import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(process.env.XJ_COMPARE_SOURCE ||
    new URL('../web/js/node.js', import.meta.url), 'utf8').replace(/^import .*;\s*$/gm, '');

async function fixture() {
    let extension;
    const opened = [];
    const frames = new Map();
    let frameId = 0;
    const app = { canvas: { pointer_is_down: false }, registerExtension(value) { extension = value; } };
    const context = vm.createContext({
        app, URL, console,
        window: { location: { href: 'http://localhost:8188/' }, open: (...args) => opened.push(args) },
        Image: class { complete = true; naturalWidth = 100; naturalHeight = 100; },
        requestAnimationFrame(callback) { frames.set(++frameId, callback); return frameId; },
        cancelAnimationFrame(id) { frames.delete(id); },
    });
    vm.runInContext(source, context);
    class Node {
        properties = {};
        widgets = [];
        pos = [500, 200];
        dirty = 0;
        addCustomWidget(widget) { this.widgets.push(widget); }
        setSize(size) { this.size = size; }
        setDirtyCanvas() { this.dirty++; }
        getExtraMenuOptions(_, options) { options.push({ content: 'Existing option' }); }
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'XJImagePairCompare' });
    const node = new Node();
    node.onNodeCreated();
    const widget = node.pairCompareWidget;
    widget.value = { pairs: [
        { a_url: '/view?filename=left%20one.png&type=temp&preview=webp&subfolder=a%2Fb&rand=123',
            b_url: '/view?filename=right.png&type=temp&preview=webp' },
        { a_url: '/view?filename=second-a.png', b_url: '/view?filename=second-b.png' },
    ] };
    function menu(x) {
        const options = [];
        node.getExtraMenuOptions(x === undefined ? undefined : { graph_mouse: [node.pos[0] + x, 250] }, options);
        return options;
    }
    function nextFrame() {
        const pending = [...frames.values()];
        frames.clear();
        pending.forEach(callback => callback());
    }
    return { node, widget, app, opened, frames, menu, nextFrame };
}

test('Open Image uses the larger visible area: slide left opens B, slide right opens A', async () => {
    const { node, widget, menu, opened } = await fixture();
    for (const [x, filename] of [[20, 'right.png'], [160, 'right.png'], [300, 'left one.png']]) {
        widget.pointerOverPos = [320 - x, 100]; // Menu position overrides stale hover.
        const options = menu(x);
        assert.deepEqual(options.map(option => option.content), ['Open Image', 'Existing option']);
        options[0].callback();
        const [url, target] = opened.at(-1);
        assert.equal(new URL(url).searchParams.get('filename'), filename);
        assert.equal(new URL(url).searchParams.has('preview'), false);
        assert.equal(target, '_blank');
    }
    widget.setSelected(1);
    menu(300)[0].callback();
    assert.equal(new URL(opened.at(-1)[0]).searchParams.get('filename'), 'second-a.png');
    assert.equal(node.size[0], 320);
});

test('menu captures its image before cursor or pair selection changes', async () => {
    const { widget, menu, opened } = await fixture();
    const options = menu(300);
    widget.pointerOverPos = [20, 100];
    widget.setSelected(1);
    options[0].callback();
    const url = new URL(opened[0][0]);
    assert.equal(url.searchParams.get('filename'), 'left one.png');
    assert.equal(url.searchParams.get('subfolder'), 'a/b');
    assert.equal(url.searchParams.get('type'), 'temp');
    assert.equal(url.searchParams.get('rand'), '123');
});

test('hover fallback and empty results produce valid menus', async () => {
    const { widget, menu, opened } = await fixture();
    widget.pointerOverPos = [300, 100];
    menu()[0].callback();
    assert.equal(new URL(opened[0][0]).searchParams.get('filename'), 'left one.png');
    widget.value = { pairs: [] };
    assert.deepEqual(menu(20).map(option => option.content), ['Existing option']);
});

test('Click mode opens its fully visible image regardless of cursor half', async () => {
    const { node, widget, menu, opened } = await fixture();
    node.properties.comparer_mode = 'Click';
    for (const x of [20, 300]) {
        widget.isPointerDown = false;
        menu(x)[0].callback();
        assert.equal(new URL(opened.at(-1)[0]).searchParams.get('filename'), 'left one.png');
        widget.isPointerDown = true;
        menu(x)[0].callback();
        assert.equal(new URL(opened.at(-1)[0]).searchParams.get('filename'), 'right.png');
    }
});

test('Slide drawing and menu agree on the majority image for a centered narrow preview', async () => {
    const { node, widget, menu, opened } = await fixture();
    widget.imagesA[0].naturalWidth = widget.imagesB[0].naturalWidth = 50;
    widget.isPointerOver = true;
    for (const x of [0, 120, 200, 320]) {
        widget.pointerOverPos = [x, 150];
        const ctx = drawingContext();
        widget.drawImages(ctx, node, 320, 0);
        const totalWidth = ctx.drawn[0][3];
        const widthB = ctx.drawn[1]?.[7] || 0;
        const majority = widthB > totalWidth / 2 ? 'right.png' : 'left one.png';
        menu(x)[0].callback();
        assert.equal(new URL(opened.at(-1)[0]).searchParams.get('filename'), majority);
    }
});

test('stationary primary press and release update Click mode and stop monitoring', async () => {
    const { node, widget, app, frames, nextFrame } = await fixture();
    app.canvas.pointer_is_down = true;
    node.onMouseDown({ button: 0 }, [40, 150], app.canvas);
    assert.equal(widget.isPointerDown, true);
    assert.equal(frames.size, 1);
    nextFrame();
    assert.equal(widget.isPointerDown, true);
    assert.equal(frames.size, 1);
    app.canvas.pointer_is_down = false; // Release outside widget, no pointerup callback.
    nextFrame();
    assert.equal(widget.isPointerDown, false);
    assert.equal(frames.size, 0);
    app.canvas.pointer_is_down = true;
    widget.mouse({ type: 'pointerdown', button: 0 }, [40, 150], node);
    assert.equal(widget.isPointerDown, true);
    widget.mouse({ type: 'pointerup', button: 0 }, [40, 150], node);
    assert.equal(widget.isPointerDown, false);
    assert.equal(frames.size, 0);
});

test('right-click does not trigger Click mode; leave and removal clear a held press', async () => {
    const { node, widget, app, frames } = await fixture();
    app.canvas.pointer_is_down = true;
    node.onMouseDown({ button: 2 }, [300, 150], app.canvas);
    assert.equal(widget.isPointerDown, false);
    node.onMouseDown({ button: 0 }, [30, 150], app.canvas);
    node.onMouseLeave({});
    assert.equal(widget.isPointerDown, false);
    assert.equal(frames.size, 0);
    node.onMouseDown({ button: 0 }, [30, 150], app.canvas);
    widget.onRemove();
    assert.equal(frames.size, 0);
    assert.equal(widget.isPointerDown, false);
});

test('saved selection restores and shorter execution results reset to a valid pair', async () => {
    const { node, widget } = await fixture();
    widget.setSelected(1);
    const saved = JSON.parse(JSON.stringify(widget.serializeValue()));
    widget.value = saved;
    assert.equal(widget.selectedPairIndex, 1);
    widget.value = { ...saved, selectedIndex: 999 };
    assert.equal(widget.selectedPairIndex, 1);
    node.onExecuted({ a_images: [{ filename: 'new-a.png', type: 'temp' }],
        b_images: [{ filename: 'new-b.png', type: 'temp' }] });
    assert.equal(widget.selectedPairIndex, 0);
    assert.equal(widget.pairCount, 1);
    assert.match(widget.imagesA[0].src, /new-a/);
});

test('missing image URLs do not shift subsequent pairs', async () => {
    const { widget } = await fixture();
    widget.value = { pairs: [{ b_url: '/b0.png' }, { a_url: '/a1.png', b_url: '/b1.png' }] };
    assert.equal(widget.imagesA[0], undefined);
    assert.equal(widget.imagesA[1].src, '/a1.png');
    assert.equal(widget.imagesB[1].src, '/b1.png');
});

function drawingContext() {
    const drawn = [];
    const context = new Proxy({ drawImage: (...args) => drawn.push(args), drawn }, {
        get: (target, name) => target[name] ?? (() => {}),
    });
    return context;
}

test('sliding right reveals A and sliding left reveals B, including letterbox edges', async () => {
    const { widget, node } = await fixture();
    widget.imagesA[0].naturalWidth = widget.imagesB[0].naturalWidth = 50;
    widget.isPointerOver = true;
    for (const [x, count] of [[0, 2], [160, 2], [320, 1]]) {
        const ctx = drawingContext();
        widget.pointerOverPos = [x, 150];
        widget.drawImages(ctx, node, 320, 0);
        assert.equal(ctx.drawn.length, count);
        if (x === 0) assert.equal(ctx.drawn[1][3], 50);
        if (x === 160) assert.ok(Math.abs(ctx.drawn[1][3] - 25) < 1e-9);
    }
});

test('failed B images and nodes too short for preview do not draw invalid images', async () => {
    const { widget, node } = await fixture();
    let ctx = drawingContext();
    widget.imagesB[0].naturalWidth = 0;
    widget.drawImages(ctx, node, 320, 0);
    assert.equal(ctx.drawn.length, 0);
    widget.imagesB[0].naturalWidth = 100;
    node.size[1] = 20;
    ctx = drawingContext();
    widget.drawImages(ctx, node, 320, 0);
    assert.equal(ctx.drawn.length, 0);
});
