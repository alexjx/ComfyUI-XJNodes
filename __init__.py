from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import os
import folder_paths
from aiohttp import web
import server
import json
from PIL import Image

WEB_DIRECTORY = "./web"


# Metadata keys for rating and comments
RATING_KEY = "xj_rating"
COMMENT_KEY = "xj_comment"


def _get_image_path(directory, subdirectory, filename):
    """Helper to construct and validate image path."""
    if directory == "input":
        base_dir = folder_paths.get_input_directory()
    elif directory == "output":
        base_dir = folder_paths.get_output_directory()
    else:
        return None

    subdirectory = subdirectory.strip().strip("/")

    if subdirectory:
        image_path = os.path.join(base_dir, subdirectory, filename)
    else:
        image_path = os.path.join(base_dir, filename)

    # Security check
    if not os.path.abspath(image_path).startswith(os.path.abspath(base_dir)):
        return None

    return image_path


def _read_rating_metadata(image_path):
    """Extract xj_rating and xj_comment from image metadata."""
    rating = None
    comment = None

    try:
        img = Image.open(image_path)
        ext = os.path.splitext(image_path)[1].lower()

        if ext in [".png", ".webp"]:
            # PNG/WebP: use text chunks
            if hasattr(img, "text") and img.text:
                rating = img.text.get(RATING_KEY)
                comment = img.text.get(COMMENT_KEY)
        elif ext in [".jpg", ".jpeg"]:
            # JPEG: use EXIF UserComment
            try:
                import piexif

                exif_dict = piexif.load(img.info.get("exif", b""))
                user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                if user_comment:
                    try:
                        # Try to parse as JSON
                        if isinstance(user_comment, bytes):
                            user_comment = user_comment.decode("utf-8")
                        data = json.loads(user_comment)
                        rating = data.get("rating")
                        comment = data.get("comment")
                    except (json.JSONDecodeError, TypeError):
                        # Legacy format: plain text comment
                        comment = user_comment
            except ImportError:
                # Fallback: try _getexif
                try:
                    exif = img._getexif()
                    if exif:
                        user_comment = exif.get(37510)
                        if user_comment:
                            try:
                                data = json.loads(user_comment)
                                rating = data.get("rating")
                                comment = data.get("comment")
                            except (json.JSONDecodeError, TypeError):
                                comment = user_comment
                except Exception:
                    pass
            except Exception:
                pass

        img.close()
    except Exception:
        pass

    # Convert rating to int if possible
    if rating is not None:
        try:
            rating = int(rating)
        except (ValueError, TypeError):
            rating = None

    return rating, comment


def _write_rating_metadata(image_path, rating, comment):
    """Write xj_rating and xj_comment to image metadata."""
    ext = os.path.splitext(image_path)[1].lower()

    # Create temporary file path
    temp_path = image_path + ".tmp"

    try:
        img = Image.open(image_path)

        if ext == ".png":
            # PNG: use text chunks
            from PIL.PngImagePlugin import PngInfo

            # Create PngInfo object
            pnginfo = PngInfo()

            # Copy existing metadata
            if hasattr(img, "text") and img.text:
                for key, value in img.text.items():
                    pnginfo.add_text(key, str(value))

            # Update with rating/comment
            if rating is not None:
                pnginfo.add_text(RATING_KEY, str(rating))
            if comment:
                pnginfo.add_text(COMMENT_KEY, comment)

            # Save with metadata
            img.save(temp_path, "PNG", pnginfo=pnginfo)

        elif ext in [".jpg", ".jpeg"]:
            # JPEG: use EXIF UserComment tag
            # Build rating data as JSON
            data = {}
            if rating is not None:
                data["rating"] = rating
            if comment:
                data["comment"] = comment

            # Use piexif for reliable EXIF handling
            try:
                import piexif

                # Load existing EXIF or create new
                try:
                    existing_exif = piexif.load(img.info.get("exif", b""))
                except Exception:
                    existing_exif = {"0th": {}, "1st": {}, "Exif": {}, "GPS": {}, "Interop": {}}

                # Ensure Exif dict exists
                if "Exif" not in existing_exif:
                    existing_exif["Exif"] = {}

                # Update UserComment (tag 37510)
                if data:
                    existing_exif["Exif"][piexif.ExifIFD.UserComment] = json.dumps(data, ensure_ascii=False).encode("utf-8")
                else:
                    existing_exif["Exif"].pop(piexif.ExifIFD.UserComment, None)

                # Dump to bytes
                exif_bytes = piexif.dump(existing_exif)
                img.save(temp_path, "JPEG", exif=exif_bytes, quality=95)

            except ImportError:
                # Fallback: try to preserve existing EXIF bytes if possible
                existing_exif_bytes = img.info.get("exif")
                if existing_exif_bytes:
                    img.save(temp_path, "JPEG", exif=existing_exif_bytes, quality=95)
                else:
                    img.save(temp_path, "JPEG", quality=95)

        elif ext == ".webp":
            # WebP: use EXIF if piexif available, otherwise just save without rating
            try:
                import piexif

                # Load existing EXIF or create new
                try:
                    existing_exif = piexif.load(img.info.get("exif", b""))
                except Exception:
                    existing_exif = {"0th": {}, "1st": {}, "Exif": {}, "GPS": {}, "Interop": {}}

                # Ensure Exif dict exists
                if "Exif" not in existing_exif:
                    existing_exif["Exif"] = {}

                # Build rating data as JSON
                data = {}
                if rating is not None:
                    data["rating"] = rating
                if comment:
                    data["comment"] = comment

                # Update UserComment (tag 37510)
                if data:
                    existing_exif["Exif"][piexif.ExifIFD.UserComment] = json.dumps(data, ensure_ascii=False).encode("utf-8")
                else:
                    existing_exif["Exif"].pop(piexif.ExifIFD.UserComment, None)

                # Dump to bytes
                exif_bytes = piexif.dump(existing_exif)
                img.save(temp_path, "WEBP", exif=exif_bytes, quality=95)

            except ImportError:
                # Fallback: save without rating metadata
                existing_exif_bytes = img.info.get("exif")
                if existing_exif_bytes:
                    img.save(temp_path, "WEBP", exif=existing_exif_bytes, quality=95)
                else:
                    img.save(temp_path, "WEBP", quality=95)

        else:
            img.close()
            return False, f"Unsupported file format: {ext}"

        img.close()

        # Replace original with temp file
        os.replace(temp_path, image_path)
        return True, None

    except Exception as e:
        # Clean up temp file if exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, str(e)


class WorkflowMetadataParser:
    """
    Extracts key parameters organized by node instance.
    Each node shows its own relevant parameters.
    """

    # Key parameters we care about (NO width/height/batch_size/seed/vae)
    KEY_PARAMS = {
        # Sampling
        "sampler": ["sampler_name", "sampler", "sampler_type"],
        "scheduler": ["scheduler", "scheduler_name"],
        "steps": ["steps", "sampling_steps", "num_steps"],
        "cfg": ["cfg", "cfg_scale", "guidance_scale"],
        "denoise": ["denoise", "denoise_strength"],
        # Models
        "checkpoint": ["ckpt_name", "model_name", "checkpoint"],
        "unet": ["unet_name"],  # Load Diffusion Model node
        "lora": ["lora_name", "lora"],
        "lora_strength": ["strength_model", "lora_strength", "strength"],
        # Prompts
        "positive": ["text", "prompt", "positive"],
        "negative": ["text", "prompt", "negative"],
    }

    def parse(self, metadata):
        """
        Returns list of nodes with their extracted parameters.
        """
        result = {"nodes": []}

        workflow = metadata.get("prompt", {})
        if not workflow:
            return result

        # Get workflow metadata for titles and node modes
        workflow_meta = metadata.get("workflow", {})
        node_titles = self._extract_node_titles(workflow_meta)
        node_modes = self._extract_node_modes(workflow_meta)

        # Process each node and extract relevant params
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})

            # Skip bypassed (mode=2) and muted (mode=4) nodes
            node_mode = node_modes.get(node_id, 0)
            if node_mode in [2, 4]:
                continue

            # Extract all key params this node has
            params = self._extract_node_params(inputs, class_type)

            # Only include nodes that have at least one key param
            if params:
                node_info = {
                    "id": node_id,
                    "type": class_type,
                    "title": node_titles.get(node_id, class_type),
                    "icon": self._get_node_icon(class_type),
                    "params": params,
                }
                result["nodes"].append(node_info)

        # Sort nodes by ID (execution order approximation)
        result["nodes"].sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 999)

        return result

    def _extract_node_params(self, inputs, class_type):
        """
        Extract all key parameters from a node's inputs.
        """
        params = {}

        for param_name, patterns in self.KEY_PARAMS.items():
            value = self._find_param_in_inputs(inputs, patterns)

            if value is not None:
                # Special handling for prompts based on class type
                if param_name in ["positive", "negative"]:
                    # Only extract if class type suggests it's a prompt node
                    if (
                        "CLIP" in class_type
                        or "Prompt" in class_type
                        or "Text" in class_type
                    ):
                        # Determine if positive or negative by class name or title
                        if "negative" in class_type.lower():
                            params["negative"] = value
                        else:
                            params["positive"] = value
                else:
                    params[param_name] = value

        return params

    def _find_param_in_inputs(self, inputs, patterns):
        """
        Find a parameter value by checking multiple name patterns.
        """
        for key, value in inputs.items():
            if any(pattern.lower() == key.lower() for pattern in patterns):
                # Handle nested arrays/lists
                if isinstance(value, list) and len(value) > 0:
                    return value[0]
                return value
        return None

    def _extract_node_titles(self, workflow_meta):
        """
        Extract custom node titles from workflow metadata.
        """
        titles = {}

        if not workflow_meta or "nodes" not in workflow_meta:
            return titles

        for node in workflow_meta.get("nodes", []):
            node_id = str(node.get("id", ""))
            title = node.get("title")
            if node_id and title:
                titles[node_id] = title

        return titles

    def _extract_node_modes(self, workflow_meta):
        """
        Extract node modes from workflow metadata.
        Mode 0 = ALWAYS (normal), Mode 2 = BYPASS, Mode 4 = MUTE
        """
        modes = {}

        if not workflow_meta or "nodes" not in workflow_meta:
            return modes

        for node in workflow_meta.get("nodes", []):
            node_id = str(node.get("id", ""))
            mode = node.get("mode", 0)
            if node_id:
                modes[node_id] = mode

        return modes

    def _get_node_icon(self, class_type):
        """
        Get emoji icon based on node type.
        """
        type_lower = class_type.lower()

        if "sampler" in type_lower:
            return "🎨"
        elif (
            "checkpoint" in type_lower or "unet" in type_lower or "model" in type_lower
        ):
            return "🤖"
        elif "lora" in type_lower:
            return "🔧"
        elif "vae" in type_lower:
            return "🎭"
        elif "clip" in type_lower or "prompt" in type_lower or "text" in type_lower:
            return "📝"
        elif "latent" in type_lower or "image" in type_lower:
            return "📐"
        else:
            return "⚙️"


# Add server API routes
@server.PromptServer.instance.routes.get("/xjnodes/get_text_file_lines")
async def get_text_file_lines(request):
    """Get the number of valid lines in a text file from the input directory"""
    filename = request.rel_url.query.get("filename", "")

    if not filename:
        return web.json_response({"error": "No filename provided"}, status=400)

    try:
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, filename)

        # Security check: ensure the file is within input directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(input_dir)):
            return web.json_response({"error": "Invalid file path"}, status=400)

        # Check file exists and has valid extension
        if not os.path.exists(file_path):
            return web.json_response({"error": "File not found"}, status=404)

        if not (filename.lower().endswith(".txt") or filename.lower().endswith(".md")):
            return web.json_response({"error": "Invalid file type"}, status=400)

        # Read and count valid lines (same logic as the node)
        with open(file_path, "r", encoding="utf-8") as f:
            text_list = f.read().splitlines()

        # Remove empty lines and comments
        text_list = [text.strip().lstrip("- ") for text in text_list]
        text_list = [text for text in text_list if text and not text.startswith("#")]

        line_count = len(text_list)

        return web.json_response(
            {
                "filename": filename,
                "line_count": line_count,
                "max_choice": max(1, line_count),
            }
        )

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/xjnodes/list_images")
async def list_images(request):
    """List image files in a directory"""
    directory = request.rel_url.query.get("directory", "input")
    subdirectory = request.rel_url.query.get("subdirectory", "")

    try:
        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        elif directory == "output":
            base_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": "Invalid directory type"}, status=400)

        # Clean subdirectory: remove leading/trailing slashes
        subdirectory = subdirectory.strip().strip("/")

        # Construct full directory path
        if subdirectory:
            full_dir = os.path.join(base_dir, subdirectory)
        else:
            full_dir = base_dir

        # Security check: ensure the path is within base directory
        if not os.path.abspath(full_dir).startswith(os.path.abspath(base_dir)):
            return web.json_response({"error": "Invalid path"}, status=400)

        # Check directory exists
        if not os.path.exists(full_dir):
            return web.json_response({"images": []})

        # List image files
        image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]
        files = []

        if os.path.isdir(full_dir):
            for f in os.listdir(full_dir):
                file_path = os.path.join(full_dir, f)
                if os.path.isfile(file_path) and any(
                    f.lower().endswith(ext) for ext in image_extensions
                ):
                    files.append(f)

        files.sort()

        return web.json_response(
            {"images": files, "directory": directory, "subdirectory": subdirectory}
        )

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/xjnodes/list_subdirectories")
async def list_subdirectories(request):
    """List subdirectories based on current path for incremental path building"""
    directory = request.rel_url.query.get("directory", "input")
    current_path = request.rel_url.query.get("current_path", "")

    try:
        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        elif directory == "output":
            base_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": "Invalid directory type"}, status=400)

        # Clean current_path: remove leading/trailing slashes
        current_path = current_path.strip().strip("/")

        # Construct full directory path to search
        if current_path:
            search_dir = os.path.join(base_dir, current_path)
        else:
            search_dir = base_dir

        # Security check: ensure the path is within base directory
        if not os.path.abspath(search_dir).startswith(os.path.abspath(base_dir)):
            return web.json_response({"error": "Invalid path"}, status=400)

        # List subdirectories
        subdirs = []

        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isdir(item_path):
                    subdirs.append(item)

        subdirs.sort()

        return web.json_response(
            {
                "subdirectories": subdirs,
                "directory": directory,
                "current_path": current_path,
            }
        )

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/xjnodes/get_image_metadata")
async def get_image_metadata(request):
    """
    Extract and parse metadata from an image file.
    Returns structured metadata with heuristic analysis.
    """
    directory = request.rel_url.query.get("directory", "input")
    subdirectory = request.rel_url.query.get("subdirectory", "")
    filename = request.rel_url.query.get("filename", "")

    if not filename:
        return web.json_response({"error": "No filename provided"}, status=400)

    try:
        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        elif directory == "output":
            base_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": "Invalid directory type"}, status=400)

        # Clean subdirectory
        subdirectory = subdirectory.strip().strip("/")

        # Construct full file path
        if subdirectory:
            image_path = os.path.join(base_dir, subdirectory, filename)
        else:
            image_path = os.path.join(base_dir, filename)

        # Security check: ensure the file is within base directory
        if not os.path.abspath(image_path).startswith(os.path.abspath(base_dir)):
            return web.json_response({"error": "Invalid file path"}, status=400)

        # Check file exists
        if not os.path.exists(image_path):
            return web.json_response({"error": "File not found"}, status=404)

        # Extract metadata from image
        raw_metadata = {}
        try:
            img = Image.open(image_path)

            # Extract PNG metadata
            if hasattr(img, "text") and img.text:
                for key, value in img.text.items():
                    try:
                        # Try to parse as JSON
                        raw_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # Keep as string if not valid JSON
                        raw_metadata[key] = value

            img.close()
        except Exception as e:
            return web.json_response(
                {"success": False, "error": f"Failed to read image metadata: {str(e)}"},
                status=500,
            )

        # Parse metadata with heuristic parser
        parser = WorkflowMetadataParser()
        parsed = parser.parse(raw_metadata)

        # Extract rating/comment from image metadata
        rating, comment = _read_rating_metadata(image_path)

        return web.json_response(
            {
                "success": True,
                "filename": filename,
                "raw_metadata": raw_metadata,
                "parsed": parsed,
                "rating": rating,
                "comment": comment,
            }
        )

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.post("/xjnodes/save_rating")
async def save_rating(request):
    """
    Save rating and comment to an image file's metadata.
    Stores xj_rating (1-5) and xj_comment in the image metadata.
    """
    try:
        data = await request.json()
        directory = data.get("directory", "input")
        subdirectory = data.get("subdirectory", "")
        filename = data.get("filename", "")
        rating = data.get("rating")
        comment = data.get("comment", "")

        if not filename:
            return web.json_response(
                {"success": False, "error": "No filename provided"}, status=400
            )

        # Validate rating
        if rating is not None:
            try:
                rating = int(rating)
                if rating < 1 or rating > 5:
                    return web.json_response(
                        {"success": False, "error": "Rating must be between 1 and 5"},
                        status=400,
                    )
            except (ValueError, TypeError):
                return web.json_response(
                    {"success": False, "error": "Invalid rating value"}, status=400
                )

        # Get and validate image path
        image_path = _get_image_path(directory, subdirectory, filename)
        if not image_path:
            return web.json_response(
                {"success": False, "error": "Invalid directory type"}, status=400
            )

        # Check file exists
        if not os.path.exists(image_path):
            return web.json_response(
                {"success": False, "error": "File not found"}, status=404
            )

        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            return web.json_response(
                {
                    "success": False,
                    "error": f"Unsupported file format: {ext}. Only PNG, JPEG, and WebP are supported.",
                },
                status=400,
            )

        # Write metadata
        success, error = _write_rating_metadata(image_path, rating, comment)

        if success:
            return web.json_response(
                {
                    "success": True,
                    "message": "Rating saved successfully",
                    "rating": rating,
                    "comment": comment,
                }
            )
        else:
            return web.json_response(
                {"success": False, "error": error or "Failed to save rating"}, status=500
            )

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


@server.PromptServer.instance.routes.get("/xjnodes/get_loras")
async def get_loras(request):
    """
    Get list of available LoRA files.
    """
    try:
        loras = folder_paths.get_filename_list("loras")
        return web.json_response({"success": True, "loras": loras})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e), "loras": []})


@server.PromptServer.instance.routes.post("/xjnodes/delete_image")
async def delete_image(request):
    """
    Delete an image file from a directory.
    """
    try:
        data = await request.json()
        directory = data.get("directory", "input")
        subdirectory = data.get("subdirectory", "")
        filename = data.get("filename", "")

        if not filename:
            return web.json_response({"error": "No filename provided"}, status=400)

        # Get base directory
        if directory == "input":
            base_dir = folder_paths.get_input_directory()
        elif directory == "output":
            base_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": "Invalid directory type"}, status=400)

        # Clean subdirectory
        subdirectory = subdirectory.strip().strip("/")

        # Construct full file path
        if subdirectory:
            image_path = os.path.join(base_dir, subdirectory, filename)
        else:
            image_path = os.path.join(base_dir, filename)

        # Security check: ensure the file is within base directory
        if not os.path.abspath(image_path).startswith(os.path.abspath(base_dir)):
            return web.json_response({"error": "Invalid file path"}, status=400)

        # Check file exists
        if not os.path.exists(image_path):
            return web.json_response({"error": "File not found"}, status=404)

        # Delete the file
        os.remove(image_path)

        return web.json_response(
            {"success": True, "message": f"Successfully deleted {filename}"}
        )

    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
