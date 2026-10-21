import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL('../web/js/random_image_from_dir.js', import.meta.url), 'utf8').replace(/^import .*;\s*$/gm, '');

test('folder browser uses selected directory and updates the subdirectory', async () => {
    let extension, menu, request, changed = 0;
    const app = { registerExtension(value) { extension = value; } };
    const api = { async fetchApi(url) { request = url; return { ok: true, json: async () => ({ subdirectories: ['child folder'] }) }; } };
    const LiteGraph = { ContextMenu: class { constructor(items, options) { menu = { items, options }; } } };
    new Function('app', 'api', 'LiteGraph', source)(app, api, LiteGraph);
    class Node {
        widgets = [{ name: 'directory', value: 'output' }, { name: 'subdirectory', value: 'parent' }];
        graph = { change() { changed++; } };
        setDirtyCanvas() {}
        addWidget(type, name, value, callback, options) { this.widgets.push({ type, name, value, callback, options }); }
    }
    await extension.beforeRegisterNodeDef(Node, { name: 'XJRandomImageFromDir' });
    const node = new Node();
    node.onNodeCreated();
    const button = node.widgets.at(-1);
    assert.equal(button.name, 'Browse Subdirs');
    assert.equal(button.options.serialize, false);
    await button.callback();
    assert.match(request, /directory=output&current_path=parent/);
    assert.equal(menu.items[1].value, '');
    menu.options.callback(menu.items[2]);
    assert.equal(node.widgets[1].value, 'parent/child folder');
    assert.equal(changed, 1);
    menu.options.callback(menu.items[0]);
    assert.equal(node.widgets[1].value, '');
});
