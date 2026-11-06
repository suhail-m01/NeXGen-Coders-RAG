// Component render smoke test: `npm run test:render`.
//
// Renders every panel with react-dom/server, so an undefined variable, a missing
// export, a bad import or a broken JSX tree fails here in half a second instead of
// in somebody's browser as a blank white page. Effects do not run during a server
// render, so this covers the render path only -- the API calls behind it are
// covered by `python scripts/apitest.py`, and the two together are what makes a
// front-end change safe to ship without clicking through it by hand.
//
// Cases may carry `expect`: a substring (or list of them) the rendered HTML must
// contain, or '' to assert that nothing is rendered at all. Length alone only
// proves something came back; these prove the data reached the screen, which is the
// failure mode a stateless component actually has -- a queue that renders an empty
// list, a drop zone that hides the formats it accepts, a refusal with no reason
// attached.
//
// Needs Node and the dev dependencies (`npm install`); it is not part of the
// Windows runtime, which serves the prebuilt bundle in dist/ and needs no Node.
import React from 'react'
import { renderToString } from 'react-dom/server'

import App from './src/App.jsx'
import Login from './src/components/Login.jsx'
import Ask from './src/components/Ask.jsx'
import Answer from './src/components/Answer.jsx'
import Evidence from './src/components/Evidence.jsx'
import StageTrace from './src/components/StageTrace.jsx'
import Sources from './src/components/Sources.jsx'
import Collections from './src/components/Collections.jsx'
import History from './src/components/History.jsx'
import Health from './src/components/Health.jsx'
import Overview from './src/components/Overview.jsx'
import DropZone from './src/components/DropZone.jsx'
import FileQueue from './src/components/FileQueue.jsx'
import Shortcuts from './src/components/Shortcuts.jsx'

const can = () => true
const cannot = () => false

const result = {
  answer: 'Budget utilisation closed at 87 percent [1]. Evidence warning: 12 was not found.',
  abstained: false,
  confidence: 0.83,
  error_code: null,
  citations: [{ citation_id: 1, item_id: 'i1', source_id: 's1', filename: 'project_report.pdf',
                modality: 'TEXT', location: 'p.3 · Budget', excerpt: 'Budget utilisation ... 87 percent' }],
  retrieved_items: [{ rank: 1, retrieval_score: 0.71, rerank_score: 0.62,
    item: { item_id: 'i1', source_id: 's1', modality: 'TEXT', content: 'Budget utilisation closed at 87 percent.',
            location: { page_number: 3, section: 'Budget' }, parent_id: null, vector_id: 4,
            metadata: { ocr_confidence: 90.5 } } }],
  relationships: [{ relationship_id: 'r1', source_item_id: 'i1', target_item_id: 'i2',
                    relationship_type: 'same_source', confidence: 1.0, metadata: {} }],
  query_representation: 'budget utilisation',
  latency_ms: { total_ms: 11000, retrieval_ms: 300, generation_ms: 9000 },
  stages: [
    { stage: 'retrieval', label: 'Hybrid retrieval', ms: 300, ran: true, enabled: true, note: '10 candidates' },
    { stage: 'rerank', label: 'Rerank', ms: 90, ran: true, enabled: true, note: '5 kept' },
    { stage: 'coverage', label: 'Coverage guard', ms: 2, ran: true, enabled: true, note: 'passed' },
    { stage: 'total', label: 'Total', ms: 11000, ran: true, enabled: true, note: '' },
    { stage: 'decision', label: 'Decision', ms: 0, ran: true, enabled: true, note: 'answered' },
  ],
  technical_detail: '',
  session_id: 'abcdef1234567890',
}

const formats = {
  documents: ['.csv', '.doc', '.docx', '.md', '.pdf', '.txt'],
  images: ['.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'],
  audio: ['.flac', '.m4a', '.mp3', '.ogg', '.wav'],
  formats: 18,
  max_file_bytes: 209715200,
}

const queue = [
  { id: 1, name: 'project_report.pdf', size: 248000, status: 'indexed', items: 4 },
  { id: 2, name: 'slide_deck.png', size: 902000, status: 'warning', items: 1,
    warnings: ['vision model unavailable; fell back to OCR (confidence 90)'] },
  { id: 3, name: 'meeting.wav', size: 1800000, status: 'uploading' },
  { id: 4, name: 'budget.csv', size: 512, status: 'waiting', waiting: 12 },
  { id: 5, name: 'locked.pdf', size: 61000, status: 'failed', error: 'the PDF is password protected' },
  { id: 6, name: 'notes.rtf', size: null, status: 'refused', error: '.rtf is not a format this index accepts' },
]

const cases = [
  ['App', <App />],
  ['Login', <Login onLogin={() => {}} />],
  ['Ask', <Ask can={can} role="admin" focusSignal={1} />],
  ['Answer', <Answer result={result} active={null} onSelect={() => {}} can={can} />],
  ['Evidence', <Evidence result={result} active={1} onSelect={() => {}} />],
  ['StageTrace', <StageTrace result={result} />],
  ['Sources', <Sources can={can} />, 'Choose files'],
  ['Sources (viewer)', <Sources can={cannot} />],
  ['DropZone', <DropZone accepted={formats} maxBytes={formats.max_file_bytes} onFiles={() => {}} />,
    'csv, doc, docx'],
  ['DropZone (no manifest)', <DropZone accepted={null} onFiles={() => {}} />,
    'Reading the accepted formats'],
  ['DropZone (disabled)', <DropZone accepted={formats} onFiles={() => {}} disabled />, 'Choose files'],
  ['FileQueue', <FileQueue queue={queue} uploading collection="Board pack" onIngest={() => {}}
      onRemove={() => {}} onClear={() => {}} />, 'the PDF is password protected'],
  ['FileQueue (waiting)', <FileQueue queue={queue} uploading={false} collection="general"
      onIngest={() => {}} onRemove={() => {}} onClear={() => {}} />,
    ['asked us to wait 12s', 'waiting on the rate limit', 'Clear finished']],
  ['FileQueue (empty)', <FileQueue queue={[]} uploading={false} onIngest={() => {}}
      onRemove={() => {}} onClear={() => {}} />, ''],
  ['Collections', <Collections can={can} sources={[{ source_id: 's1', filename: 'a.pdf',
      source_type: 'pdf', collection: 'general' }]} onChanged={() => {}} />],
  ['Collections (viewer)', <Collections can={cannot} sources={[]} onChanged={() => {}} />],
  ['History', <History can={can} />],
  ['History (viewer)', <History can={cannot} />],
  ['Health', <Health />],
  // Overview renders its loading state under renderToString: every figure on it
  // comes from an endpoint, and effects do not run on the server. What this
  // asserts is that it does not crash while waiting and does not invent a number
  // to fill the gap - the rule for that component is an em dash, never a
  // plausible constant.
  ['Overview', <Overview onNavigate={() => {}} />, ['Reading service state']],
  // App cannot be rendered signed in (its effects do not run on the server), so the
  // overlay is a component of its own and gets covered here instead.
  ['Shortcuts', <Shortcuts tabs={[{ id: 'ask', label: 'Ask' }, { id: 'sources', label: 'Sources' }]}
      onClose={() => {}} />, ['Keyboard shortcuts', 'Jump to the question box', 'Sources']],
]

let failures = 0
for (const [name, element, expect] of cases) {
  try {
    const raw = renderToString(element)
    // Server rendering separates adjacent text and expressions with <!-- -->.
    // Stripping it lets an expectation be written the way a person reads the panel.
    const html = raw.replace(/<!--\s*-->/g, '')
    if (expect === '') {
      if (raw !== '') throw new Error(`expected nothing rendered, got ${raw.length} characters`)
    } else {
      if (!html || html.length < 20) throw new Error(`rendered ${html.length} characters`)
      for (const needle of expect ? [].concat(expect) : []) {
        if (!html.includes(needle)) {
          throw new Error(`rendered, but the output has no "${needle}"`)
        }
      }
    }
    console.log(`ok    ${name.padEnd(22)} ${html.length} chars`)
  } catch (err) {
    failures += 1
    console.log(`FAIL  ${name.padEnd(22)} ${err.message}`)
  }
}
console.log(
  failures
    ? `\n${failures} of ${cases.length} component(s) failed to render`
    : `\nall ${cases.length} components rendered`,
)
process.exit(failures ? 1 : 0)
