import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

const available = spawnSync('agent-browser', ['--version'], { encoding: 'utf8' }).status === 0;
const source = readFileSync(new URL('../web/js/node.js', import.meta.url), 'utf8')
    .replace(/^import .*;\s*$/gm, '');
const session = `xj-text-list-test-${process.pid}`;

function browser(args, input) {
    const result = spawnSync('agent-browser', ['--session', session, '--args', '--no-sandbox', '--json', ...args], {
        encoding: 'utf8', input, timeout: 30_000, maxBuffer: 2 * 1024 * 1024,
    });
    assert.equal(result.status, 0, result.stderr || result.stdout || result.error?.message);
    const response = JSON.parse(result.stdout);
    assert.equal(response.success, true, JSON.stringify(response));
    return response.data?.result;
}

function evaluate(script) {
    return browser(['eval', '--stdin'], `(async () => { ${script} })()`);
}

// This optional integration test exercises native layout and ResizeObserver,
// which pure parser tests cannot cover. Run with: node --test tests/*.mjs
test('Text From List browser lifecycle and interactions', {
    skip: !available && 'agent-browser is not installed', timeout: 90_000,
}, async (t) => {
    try {
        browser(['open', 'about:blank']);
        evaluate(`
            document.body.style.margin = '0';
            let extension;
            new Function('app', ${JSON.stringify(source)})({ registerExtension(value) { extension = value; } });
            window.settle = async () => {
                for (let i = 0; i < 4; i++) await new Promise(requestAnimationFrame);
            };
            window.check = (condition, message) => { if (!condition) throw new Error(message); };
            window.measurements = 0;
            new MutationObserver(records => {
                for (const record of records) for (const element of record.addedNodes) {
                    if (element.nodeType === 1 && element.style.visibility === 'hidden') measurements++;
                }
            }).observe(document.body, { childList: true });
            window.createFixture = () => {
                const choice = { name: 'choice', value: 1, options: {} };
                const node = {
                    widgets: [choice], setDirtyCanvas() {},
                    onAdded() { this.additions = (this.additions || 0) + 1; },
                    addDOMWidget(name, type, element, options) {
                        const widget = { name, type, element, options,
                            onRemove() { this.removals = (this.removals || 0) + 1; } };
                        Object.defineProperty(widget, 'value', {
                            get: () => options.getValue(), set: value => options.setValue(value),
                        });
                        this.widgets.push(widget);
                        return widget;
                    },
                };
                const { widget } = extension.getCustomWidgets().XJ_NUMBERED_LIST(node, 'text');
                const host = document.createElement('div');
                host.style.cssText = 'width: 480px; height: 200px';
                host.appendChild(widget.element);
                return { node, choice, widget, host, container: widget.element,
                    gutter: widget.element.children[0], textarea: widget.element.querySelector('textarea'),
                    format: widget.element.querySelector('[title="格式化列表"]'),
                    locate: widget.element.querySelector('[title="定位 choice 条目"]') };
            };
            window.fixture = createFixture();
            return true;
        `);

        await t.test('detached and hidden widgets defer measurement until visible', () => {
            assert.equal(evaluate(`
                const f = fixture;
                f.widget.value = '- ' + 'longtext'.repeat(300) + '\\n- second';
                await settle();
                check(measurements === 0, 'detached widget measured text');
                check(f.gutter.children.length === 0, 'detached widget created gutter rows');
                f.host.style.display = 'none';
                document.body.appendChild(f.host);
                f.widget.value = '- hidden\\n-\\n- visible';
                await settle();
                check(measurements === 0, 'display:none widget measured text');
                f.host.style.display = '';
                await settle();
                check(measurements > 0, 'visible widget did not measure text');
                check(f.gutter.querySelectorAll('[data-line-number]').length === 2, 'empty marker was numbered');
                check(f.choice.options.max === 2, 'choice bounds differ from valid candidates');
                return true;
            `), true);
        });

        await t.test('same text and height changes reuse rows; width and text changes refresh', () => {
            assert.equal(evaluate(`
                const f = fixture;
                const rows = [...f.gutter.children];
                const count = measurements;
                f.widget.value = f.widget.value;
                f.host.style.height = '240px';
                await settle();
                check(rows.every((row, i) => f.gutter.children[i] === row), 'unchanged wrapping replaced rows');
                check(measurements === count, 'unchanged wrapping measured again');
                f.widget.value += '\\n- changed';
                await settle();
                check(f.gutter.firstElementChild !== rows[0], 'changed content kept stale rows');
                const refreshed = f.gutter.firstElementChild;
                f.host.style.width = '320px';
                await settle();
                check(f.gutter.firstElementChild !== refreshed, 'changed width kept stale rows');
                const countBeforeHide = measurements;
                f.host.style.display = 'none';
                await settle();
                f.widget.value = '- hidden update\\n- another update';
                await settle();
                check(measurements === countBeforeHide, 'hidden update measured text');
                f.host.style.display = '';
                await settle();
                check(f.gutter.querySelectorAll('[data-line-number]').length === 2, 'showing widget retained stale rows');
                return true;
            `), true);
        });

        await t.test('wrapped words, Chinese, tabs and continuations keep gutter aligned', () => {
            assert.equal(evaluate(`
                const f = fixture;
                const text = [
                    '# comments', '- ' + 'unbrokenword'.repeat(40),
                    '  ' + '中文长文本'.repeat(40), '\\t' + 'tabbed text '.repeat(30),
                    '', '- second', '  continuation', '- ', '# ignored', '- last',
                ].join('\\n');
                f.widget.value = text;
                await settle();
                check(f.gutter.children.length === text.split('\\n').length, 'gutter DOM grows with visual wraps');
                check(Math.abs(f.gutter.scrollHeight - f.textarea.scrollHeight) <= 2,
                    'gutter and textarea scroll heights differ: ' + f.gutter.scrollHeight + '/' + f.textarea.scrollHeight);
                check(f.choice.options.max === 3, 'empty marker changed candidate count');
                const target = f.gutter.querySelector('[data-line-number="3"]');
                f.choice.value = 3;
                f.locate.click();
                await settle();
                check(f.textarea.scrollTop > 0, 'locate did not scroll to last candidate');
                check(f.gutter.scrollTop === f.textarea.scrollTop, 'locate desynchronized gutter scroll');
                const targetTop = target.getBoundingClientRect().top;
                const viewport = f.gutter.getBoundingClientRect();
                check(targetTop >= viewport.top && targetTop < viewport.bottom, 'located candidate is outside viewport');
                f.textarea.scrollTop = 100;
                f.textarea.dispatchEvent(new Event('scroll'));
                check(f.gutter.scrollTop === f.textarea.scrollTop, 'manual scroll desynchronized gutter');
                return true;
            `), true);
        });

        await t.test('format is idempotent and preserves serialized value and choice', () => {
            assert.equal(evaluate(`
                const f = fixture;
                f.widget.value = '\\n# group\\n# detail\\n- first\\n  continuation\\n\\n\\n-\\n- second\\n';
                f.choice.value = 2;
                await settle();
                f.format.click();
                await settle();
                const expected = '# group\\n# detail\\n\\n- first\\n  continuation\\n\\n-\\n\\n- second';
                check(f.textarea.value === expected, 'format changed content unexpectedly');
                check(f.widget.value === expected && f.widget.options.getValue() === expected, 'serialized value is stale');
                check(f.choice.value === 2, 'format changed choice');
                const row = f.gutter.firstElementChild;
                const count = measurements;
                f.format.click();
                await settle();
                check(f.widget.value === expected, 'repeated format changed value');
                check(f.gutter.firstElementChild === row && measurements === count, 'repeated format rebuilt gutter');
                f.widget.value = '# comments only\\n-';
                await settle();
                check(f.locate.disabled, 'locate stays enabled without candidates');
                check(f.choice.value === 1 && f.choice.options.max === 1, 'empty list has invalid choice bounds');
                return true;
            `), true);
        });

        await t.test('contents navigates wrapped source lines without changing text or choice', () => {
            assert.equal(evaluate(`
                const f = fixture;
                const text = '# First\\n- ' + 'long text '.repeat(200) + '\\n  ## 中文 section\\n- last';
                f.widget.value = text;
                f.choice.value = 2;
                await settle();
                const toc = f.container.querySelector('[title="Table of contents"]');
                const panel = f.container.querySelector('[aria-label="Table of contents"][role="region"]');
                toc.click();
                check(!panel.hidden && toc.getAttribute('aria-expanded') === 'true', 'TOC did not open');
                check(panel.querySelectorAll('button').length === 2, 'wrong heading count');
                check(document.activeElement === panel.querySelector('button'), 'TOC did not receive focus');
                panel.querySelectorAll('button')[1].click();
                await settle();
                check(panel.hidden && document.activeElement === f.textarea, 'navigation did not return to editor');
                check(f.textarea.value.slice(f.textarea.selectionStart, f.textarea.selectionEnd) === '  ## 中文 section', 'wrong source selection');
                check(f.textarea.scrollTop > 0 && f.gutter.scrollTop === f.textarea.scrollTop, 'scroll did not synchronize');
                const row = f.gutter.children[2].getBoundingClientRect();
                const viewport = f.gutter.getBoundingClientRect();
                check(row.top >= viewport.top && row.top < viewport.bottom, 'heading outside viewport');
                check(f.widget.value === text && f.choice.value === 2, 'navigation changed data');
                toc.click();
                panel.querySelector('button').dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
                check(panel.hidden && document.activeElement === toc, 'Escape did not dismiss and restore focus');
                toc.click();
                f.widget.value = '- no headings';
                check(panel.hidden, 'external update left stale contents');
                toc.click();
                check(panel.textContent.includes('# Section title'), 'missing empty-state help');
                f.textarea.value = '# New heading\\n- item';
                f.textarea.dispatchEvent(new Event('input'));
                check(panel.hidden, 'typing left stale contents');
                toc.click();
                check(panel.querySelector('button').textContent === 'New heading · Line 1', 'contents did not refresh');
                f.textarea.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
                check(panel.hidden, 'editor click did not dismiss');
                return true;
            `), true);
        });

        await t.test('removal cancels resize and typing work and calls original cleanup', () => {
            assert.equal(evaluate(`
                const f = fixture;
                const row = f.gutter.firstElementChild;
                const count = measurements;
                f.textarea.value = '- pending typed update';
                f.textarea.dispatchEvent(new Event('input'));
                f.widget.onRemove();
                f.host.style.width = '420px';
                await new Promise(resolve => setTimeout(resolve, 350));
                await settle();
                check(f.widget.removals === 1, 'original widget cleanup was skipped');
                check(measurements === count && f.gutter.firstElementChild === row, 'removed widget still performs layout work');
                return true;
            `), true);
        });

        await t.test('readding a removed node restores resize observation and original lifecycle', () => {
            assert.equal(evaluate(`
                const f = fixture;
                f.textarea.value = '- ' + 'restored wrapping '.repeat(80) + '\\n- second';
                const count = measurements;
                f.node.onAdded();
                f.host.style.width = '260px';
                await settle();
                check(f.node.additions === 1, 'original node onAdded was skipped');
                check(measurements > count, 'readded node did not resume measurement');
                check(Math.abs(f.gutter.scrollHeight - f.textarea.scrollHeight) <= 2,
                    'readded node gutter wrapping differs from textarea');
                f.widget.onRemove();
                check(f.widget.removals === 2, 'second removal skipped original cleanup');
                return true;
            `), true);
        });
    } finally {
        browser(['close']);
    }
});
