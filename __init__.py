from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import os
import folder_paths
from aiohttp import web
import server

WEB_DIRECTORY = "./web"

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
        text_list = [
            text for text in text_list if text and not text.startswith("#")
        ]

        line_count = len(text_list)

        return web.json_response({
            "filename": filename,
            "line_count": line_count,
            "max_choice": max(1, line_count)
        })

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
        subdirectory = subdirectory.strip().strip('/')

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
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif']
        files = []

        if os.path.isdir(full_dir):
            for f in os.listdir(full_dir):
                file_path = os.path.join(full_dir, f)
                if os.path.isfile(file_path) and any(f.lower().endswith(ext) for ext in image_extensions):
                    files.append(f)

        files.sort()

        return web.json_response({
            "images": files,
            "directory": directory,
            "subdirectory": subdirectory
        })

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
