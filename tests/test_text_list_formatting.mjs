import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Run the real frontend helpers without loading ComfyUI or a DOM.
const source = readFileSync(new URL('../web/js/node.js', import.meta.url), 'utf8');
const pureFunctions = source.split('// Image Navigator Modal Class')[0]
    .replace(/^import .*;\s*$/gm, '');
const context = vm.createContext({});
vm.runInContext(pureFunctions, context);
const format = (text) => context.formatTextList(text);
const items = (text) => JSON.parse(JSON.stringify(context.processTextList(text)));

test('section headings use nonempty full-line comments and preserve source positions', () => {
    const input = '# First\r\n- item # inline\r\n  ## 中文\r\n#\r\n  #   \r\n# First\r\n- last';
    assert.deepEqual(JSON.parse(JSON.stringify(context.getTextListSections(input))), [
        { title: 'First', lineIndex: 0 },
        { title: '中文', lineIndex: 2 },
        { title: 'First', lineIndex: 5 },
    ]);
    assert.deepEqual(items(input).map(item => item.text), ['item # inline', 'last']);
});

test('format inserts exactly one boundary blank line and is idempotent', () => {
    const input = '\n\n- first\n- second\n\n\n\n- third\n\n';
    const expected = '- first\n\n- second\n\n- third';
    assert.equal(format(input), expected);
    assert.equal(format(format(input)), expected);
});

test('format groups adjacent comments and separates them from items', () => {
    const input = '\n# group\n  # detail\n- first\n\n\n# next\n# more\n- second\n';
    const expected = '# group\n  # detail\n\n- first\n\n# next\n# more\n\n- second';
    assert.equal(format(input), expected);
    assert.equal(format(expected), expected);
});

test('separate comment groups retain exactly one boundary blank line', () => {
    const input = '# first\n# detail\n\n \n\n# second\n# more\n- item';
    const expected = '# first\n# detail\n\n# second\n# more\n\n- item';
    assert.equal(format(input), expected);
    assert.equal(format(expected), expected);
});

test('format preserves multiline item and comment content, indentation, and internal blanks', () => {
    const input = '\n- first\n  continuation\n\n\n\tlast line\n\n\n# notes\n  detail\n\n  more detail\n- second\n';
    const expected = '- first\n  continuation\n\n\n\tlast line\n\n# notes\n  detail\n\n  more detail\n\n- second';
    assert.equal(format(input), expected);
    assert.equal(format(expected), expected);
});

test('empty markers remain in source and do not receive a candidate index', () => {
    const input = '-\n\n-   \n# ignored\n- first\n-\n- second';
    assert.equal(format(input), '-\n\n-   \n\n# ignored\n\n- first\n\n-\n\n- second');
    const result = items(input);
    assert.deepEqual(result.map(({ index, text, lineIndex }) => ({ index, text: text.trim(), lineIndex })), [
        { index: 1, text: 'first', lineIndex: 4 },
        { index: 2, text: 'second', lineIndex: 6 },
    ]);
    for (const item of result) assert.equal(item.originalLine, item.text);
});

test('an empty marker with continuation content is a valid candidate', () => {
    const input = '# ignored\n-\n  continued content\n\n- next';
    const result = items(input);
    assert.deepEqual(result.map(({ index, text, lineIndex }) => ({ index, text: text.trim(), lineIndex })), [
        { index: 1, text: 'continued content', lineIndex: 1 },
        { index: 2, text: 'next', lineIndex: 4 },
    ]);
    assert.equal(format(input), '# ignored\n\n-\n  continued content\n\n- next');
});

test('orphan leading content is preserved but excluded from candidate numbering', () => {
    const input = '\nintro\n  extra introduction\n\n\n- first\n# ignored\n  ignored continuation\n- second';
    assert.equal(format(input), 'intro\n  extra introduction\n\n- first\n\n# ignored\n  ignored continuation\n\n- second');
    assert.deepEqual(items(input).map(({ index, text, lineIndex }) => ({ index, text: text.trim(), lineIndex })), [
        { index: 1, text: 'first', lineIndex: 5 },
        { index: 2, text: 'second', lineIndex: 8 },
    ]);
});

test('CRLF sources normalize to LF without changing logical line positions', () => {
    const input = '\r\n# group\r\n- first\r\n  continuation\r\n\r\n\r\n- second\r\n';
    assert.equal(format(input), '# group\n\n- first\n  continuation\n\n- second');
    assert.deepEqual(items(input).map(({ index, text, lineIndex }) => ({ index, text: text.trim(), lineIndex })), [
        { index: 1, text: 'first\n  continuation', lineIndex: 2 },
        { index: 2, text: 'second', lineIndex: 6 },
    ]);
});

test('empty and comment-only lists have no candidates', () => {
    for (const input of ['', '\n \n\t\n', '# only\n# comments\n', '-\n-   \n# comment']) {
        assert.deepEqual(items(input), []);
        assert.equal(format(format(input)), format(input));
    }
    assert.equal(format('\n \n\t\n'), '');
});
