import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

app.registerExtension({
    name: 'XJNodes.RandomImageFromDir',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'XJRandomImageFromDir') return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const node = this;
            const directory = this.widgets.find(w => w.name === 'directory');
            const subdirectory = this.widgets.find(w => w.name === 'subdirectory');
            this.addWidget('button', 'Browse Subdirs', null, async (_value, _canvas, _node, event) => {
                const current = (subdirectory.value || '').trim().replace(/^\/+|\/+$/g, '');
                const params = new URLSearchParams({ directory: directory.value, current_path: current });
                try {
                    const response = await api.fetchApi(`/xjnodes/list_subdirectories?${params}`);
                    if (!response.ok) throw new Error(`Directory listing failed (${response.status})`);
                    const data = await response.json();
                    const choices = [{ content: '(Root)', value: '' }];
                    if (current) choices.push({ content: '.. (Parent Directory)', value: current.split('/').slice(0, -1).join('/') });
                    for (const name of data.subdirectories || []) {
                        choices.push({ content: name, value: current ? `${current}/${name}` : name });
                    }
                    new LiteGraph.ContextMenu(choices, {
                        event,
                        callback(choice) {
                            subdirectory.value = choice.value;
                            subdirectory.callback?.(choice.value);
                            node.graph?.change();
                            node.setDirtyCanvas(true, true);
                        },
                    });
                } catch (error) {
                    console.error('[XJNodes] Unable to browse subdirectories', error);
                    app.extensionManager?.toast?.add({ severity: 'error', summary: 'Unable to browse subdirectories', detail: error.message, life: 5000 });
                }
            }, { serialize: false });
            // ComfyUI displays the backend's ui.images response as its native preview.
            return result;
        };
    },
});
