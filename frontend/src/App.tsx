import { useEffect, useMemo, useState } from "react";
import type { Review, ReviewPayload, Source } from "./types";

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;
const score = (n: number) => n.toFixed(3);
const shortId = (id: string) => (id.length > 12 ? `${id.slice(0, 8)}…` : id);

export default function App() {
  const [sources, setSources] = useState<Source[]>([]);
  const [source, setSource] = useState("safetyhops");
  const [modelType, setModelType] = useState("logistic");
  const [limit, setLimit] = useState(20);
  const [data, setData] = useState<ReviewPayload | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/sources")
      .then((r) => r.json())
      .then((body) => {
        const list: Source[] = body.sources ?? [];
        setSources(list);
        const first = list.find((s) => s.available);
        if (first) {
          setSource(first.id);
        }
      })
      .catch(() => setError("Could not reach the SafetyNet API. Start it with: python -m uvicorn api:app --reload"));
  }, []);

  async function loadReview(nextSource = source) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(
        `/api/review?source=${encodeURIComponent(nextSource)}&model_type=${modelType}&limit=${limit}`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const payload: ReviewPayload = await res.json();
      setData(payload);
      setSelectedId(payload.reviews[0]?.case_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const selected: Review | undefined = useMemo(
    () => data?.reviews.find((row) => row.case_id === selectedId),
    [data, selectedId]
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="mark">SN</div>
          <div>
            <h1>SafetyNet</h1>
            <p>Second-look review queue for possible harm events</p>
          </div>
        </div>
        <div className="controls">
          <label>
            Data source
            <select
              value={source}
              onChange={(e) => {
                setSource(e.target.value);
              }}
            >
              {sources.map((item) => (
                <option key={item.id} value={item.id} disabled={!item.available}>
                  {item.label}
                  {!item.available ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="logistic">Logistic</option>
              <option value="tree">Decision tree</option>
            </select>
          </label>
          <label>
            Top reviews
            <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={40}>40</option>
            </select>
          </label>
          <button type="button" onClick={() => loadReview()} disabled={loading}>
            {loading ? "Scoring…" : "Load reviews"}
          </button>
        </div>
      </header>

      {error ? <div className="error">{error}</div> : null}

      <section className="metrics">
        <div className="metric">
          <span>Stays the tool checked</span>
          <strong>{data?.checked ?? "—"}</strong>
        </div>
        <div className="metric">
          <span>Top Reviews</span>
          <strong>{data?.top_reviews ?? "—"}</strong>
        </div>
        <div className="metric">
          <span>Already tagged as harm</span>
          <strong>{data ? pct(data.already_tagged_share) : "—"}</strong>
        </div>
        <div className="metric">
          <span>Tagged stays this list found</span>
          <strong>
            {data ? `${data.tagged_found} / ${data.tagged_total}` : "—"}
          </strong>
        </div>
      </section>

      <section className="layout">
        <div className="panel">
          <h2>Review list</h2>
          {!data && !loading ? (
            <div className="status">Choose a source and load reviews to fill the queue.</div>
          ) : null}
          {loading ? <div className="status">Scoring the cohort. First load can take a minute.</div> : null}
          <div className="list">
            {data?.reviews.map((row) => (
              <button
                key={row.case_id}
                className={row.case_id === selectedId ? "row active" : "row"}
                type="button"
                onClick={() => setSelectedId(row.case_id)}
              >
                <div className="row-top">
                  <span className="mono" title={row.case_id}>
                    {shortId(row.case_id)}
                  </span>
                  <span className="score">{score(row.score)}</span>
                </div>
                <div className="chips">
                  <span className={row.label ? "chip tagged" : "chip clear"}>
                    {row.label ? "already tagged" : "not tagged"}
                  </span>
                  <span className="chip">
                    {row.age} / {row.gender}
                  </span>
                  <span className="chip">{row.scenario}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Case detail</h2>
          {!selected ? (
            <div className="status">Select a stay on the left.</div>
          ) : (
            <div className="detail">
              <div>
                <div className="row-top">
                  <strong className="mono" title={selected.case_id}>
                    {shortId(selected.case_id)}
                  </strong>
                  <span className="score">{score(selected.score)}</span>
                </div>
                <p>
                  {selected.age}-year-old {selected.gender}. {selected.event_count} events on
                  file. Label: {data?.label_name}.
                </p>
              </div>
              <div className="grid-2">
                <div className="card">
                  <h3>What raised the score</h3>
                  {selected.raising.length === 0 ? <div className="status">None</div> : null}
                  {selected.raising.map((item) => (
                    <div className="item" key={item.feature}>
                      {item.description || item.feature}
                      <br />
                      <em>
                        +{item.contribution.toFixed(3)} · {item.feature}
                      </em>
                    </div>
                  ))}
                </div>
                <div className="card">
                  <h3>What lowered the score</h3>
                  {selected.lowering.length === 0 ? <div className="status">None</div> : null}
                  {selected.lowering.map((item) => (
                    <div className="item" key={item.feature}>
                      {item.description || item.feature}
                      <br />
                      <em>
                        {item.contribution.toFixed(3)} · {item.feature}
                      </em>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card">
                <h3>Watcher triggers</h3>
                {selected.triggers.length === 0 ? <div className="status">No triggers fired.</div> : null}
                {selected.triggers.map((text) => (
                  <div className="item" key={text}>
                    {text}
                  </div>
                ))}
              </div>
              <div className="card">
                <h3>Recent timeline</h3>
                {selected.timeline.map((event, idx) => (
                  <div className="item" key={`${event.timestamp}-${idx}`}>
                    <span className="mono">{event.timestamp}</span> · {event.event_type}: {event.value}
                    {event.details ? <em> — {event.details}</em> : null}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {data ? (
        <div className="panel coverage">
          <h2>Trigger coverage</h2>
          <table>
            <thead>
              <tr>
                <th>Trigger</th>
                <th>Meaning</th>
                <th>Cases</th>
                <th>Share</th>
                <th>Label rate</th>
                <th>Lift</th>
              </tr>
            </thead>
            <tbody>
              {data.coverage.slice(0, 14).map((row) => (
                <tr key={row.trigger}>
                  <td className="mono">{row.trigger}</td>
                  <td>{row.meaning}</td>
                  <td>{row.cases}</td>
                  <td>{pct(row.share)}</td>
                  <td>{row.label_rate == null ? "—" : pct(row.label_rate)}</td>
                  <td>{row.lift == null ? "—" : row.lift.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
