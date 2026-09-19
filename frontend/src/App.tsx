import { useEffect, useMemo, useState } from "react";
import type { Review, ReviewPayload, Source } from "./types";

type MainTab = "queue" | "import" | "more";

const LIST_LIMIT = 10;

const shortId = (id: string) => (id.length > 12 ? `${id.slice(0, 8)}...` : id);
const plain = (text: string) => text.replace(/\*\*/g, "");

function waitLabel(row: Pick<Review, "waiting_days" | "stale">): string | null {
  if (row.waiting_days == null) {
    return null;
  }
  const days = row.waiting_days;
  const unit = days === 1 ? "day" : "days";
  if (row.stale) {
    return `Waiting ${days} ${unit} (over a week)`;
  }
  return `Waiting ${days} ${unit}`;
}

function whyOnList(row: Review): string {
  if (row.triggers[0]) {
    return plain(row.triggers[0]);
  }
  if (row.raising[0]) {
    return plain(row.raising[0].description || row.raising[0].feature);
  }
  return "Flagged for a second look";
}

export default function App() {
  const [sources, setSources] = useState<Source[]>([]);
  const [source, setSource] = useState("safetyhops");
  const [modelType, setModelType] = useState("logistic");
  const [data, setData] = useState<ReviewPayload | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [bundleFiles, setBundleFiles] = useState<File[]>([]);
  const [importing, setImporting] = useState(false);
  const [importNote, setImportNote] = useState("");
  const [mappingNote, setMappingNote] = useState("");
  const [tab, setTab] = useState<MainTab>("queue");

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
      .catch(() =>
        setError("Could not reach the SafetyNet API. Start it with: python -m uvicorn api:app --reload")
      );
  }, []);

  async function refreshSources() {
    const res = await fetch("/api/sources");
    const body = await res.json();
    const list: Source[] = body.sources ?? [];
    setSources(list);
    return list;
  }

  async function importFiles() {
    if (bundleFiles.length === 0) {
      setError("Choose a folder of CSVs or one zip.");
      return;
    }
    setImporting(true);
    setError("");
    setImportNote("");
    setMappingNote("");
    try {
      const body = new FormData();
      for (const file of bundleFiles) {
        body.append("files", file, file.name);
      }
      const res = await fetch("/api/hospital/upload?rank=true&model_type=" + modelType, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const detail = payload.detail;
        throw new Error(
          typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `Import failed (${res.status})`
        );
      }
      const result = await res.json();
      setImportNote(
        `Stored ${result.stays_upserted} stays (${result.in_house} still in hospital, ${result.discharged} discharged). ${result.ranked.scored} new discharges added to the list.`
      );
      if (result.mapping?.files) {
        setMappingNote(
          Object.entries(result.mapping.files)
            .map(([role, name]) => `${role}: ${name}`)
            .join(" | ")
        );
      }
      await refreshSources();
      setSource("hospital");
      if ((result.discharged ?? 0) > 0) {
        await loadReview("hospital");
        setTab("queue");
      } else {
        setData(null);
        setImportNote(
          `Stored ${result.stays_upserted} stays. None have a discharge time yet, so nothing is on the review list. Import again after they are discharged.`
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Import failed";
      setError(typeof message === "string" ? message : JSON.stringify(message));
    } finally {
      setImporting(false);
    }
  }

  async function loadReview(nextSource = source) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(
        `/api/review?source=${encodeURIComponent(nextSource)}&model_type=${modelType}&limit=${LIST_LIMIT}`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const payload: ReviewPayload = await res.json();
      setData(payload);
      setSelectedId(payload.reviews[0]?.case_id ?? null);
      setTab("queue");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function markReviewed(caseId: string, decision: string, notes?: string) {
    if (!data?.can_review) {
      setError("Reviews are saved only for hospital stays stored in SafetyNet.");
      return;
    }
    setError("");
    const res = await fetch("/api/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: data.source,
        case_id: caseId,
        decision,
        reviewer: "queue",
        notes: notes ?? null,
      }),
    });
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      const detail = payload.detail;
      throw new Error(
        typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : "Could not save review"
      );
    }
    const remaining = data.reviews.filter((row) => row.case_id !== caseId);
    setData({ ...data, reviews: remaining, top_reviews: remaining.length });
    if (selectedId === caseId) {
      setSelectedId(remaining[0]?.case_id ?? null);
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
            <p>Charts that may need a second look after discharge</p>
          </div>
        </div>
      </header>

      <nav className="tabs" aria-label="Main sections">
        <button type="button" className={tab === "queue" ? "tab active" : "tab"} onClick={() => setTab("queue")}>
          Review list
        </button>
        <button type="button" className={tab === "import" ? "tab active" : "tab"} onClick={() => setTab("import")}>
          Import records
        </button>
        <button type="button" className={tab === "more" ? "tab active" : "tab"} onClick={() => setTab("more")}>
          Extra detail
        </button>
      </nav>

      {error ? <div className="error">{error}</div> : null}

      {tab === "import" ? (
        <section className="importer">
          <div>
            <h2>Bring in hospital files</h2>
            <p>
              Upload one folder of tables, or a zip. File names can be theirs. Only discharged stays
              appear on the review list.
            </p>
          </div>
          <div className="folder-pick">
            <label>
              Choose a folder
              <input
                type="file"
                multiple
                onChange={(e) => setBundleFiles(Array.from(e.target.files ?? []))}
                {...{ webkitdirectory: "", directory: "" }}
              />
            </label>
            <label>
              Or a zip / loose CSVs
              <input
                type="file"
                multiple
                accept=".csv,.zip"
                onChange={(e) => setBundleFiles(Array.from(e.target.files ?? []))}
              />
            </label>
          </div>
          <button type="button" onClick={importFiles} disabled={importing}>
            {importing ? "Importing and ranking... wait up to a minute" : "Import folder"}
          </button>
          {bundleFiles.length ? (
            <p>
              {bundleFiles.length} file{bundleFiles.length === 1 ? "" : "s"} selected
            </p>
          ) : null}
          {importNote ? <p className="import-note">{importNote}</p> : null}
          {mappingNote ? <p>{mappingNote}</p> : null}
        </section>
      ) : null}

      {tab === "more" ? (
        <section className="more-page">
          <div className="importer">
            <h2>Which records and model</h2>
            <p>Leave these alone unless you are testing. Hospital is the live list.</p>
            <div className="controls stacked">
              <label>
                Data source
                <select value={source} onChange={(e) => setSource(e.target.value)}>
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
            </div>
          </div>

          {data ? (
            <>
              <section className="metrics dense">
                <div className="metric">
                  <span>Already tagged as harm (this list)</span>
                  <strong>{`${(data.already_tagged_share * 100).toFixed(1)}%`}</strong>
                </div>
                <div className="metric">
                  <span>Tagged stays this list found</span>
                  <strong>
                    {data.tagged_found} / {data.tagged_total}
                  </strong>
                </div>
              </section>
              {data.report ? (
                <section className="metrics dense">
                  <div className="metric">
                    <span>PR AUC (held out)</span>
                    <strong>{data.report.average_precision.toFixed(3)}</strong>
                    <small>
                      {data.report.pr_lift.toFixed(2)}x the{" "}
                      {(data.report.pr_baseline * 100).toFixed(1)}% base rate
                    </small>
                  </div>
                  <div className="metric">
                    <span>PR AUC (cross-validated)</span>
                    <strong>
                      {data.report.cv_average_precision_mean.toFixed(3)} +/-{" "}
                      {data.report.cv_average_precision_std.toFixed(3)}
                    </strong>
                    <small>the number to trust when positives are scarce</small>
                  </div>
                  <div className="metric">
                    <span>ROC AUC (held out)</span>
                    <strong>{data.report.roc_auc.toFixed(3)}</strong>
                    <small>always nulls at 0.500, so it flatters a rare-event model</small>
                  </div>
                  <div className="metric">
                    <span>Label prevalence</span>
                    <strong>{`${(data.report.positive_rate * 100).toFixed(1)}%`}</strong>
                    <small>
                      {data.report.positive_rate > 0.3
                        ? "far denser than harm in a real hospital population"
                        : "also the PR AUC baseline"}
                    </small>
                  </div>
                </section>
              ) : null}
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
                        <td>{`${(row.share * 100).toFixed(1)}%`}</td>
                        <td>{row.label_rate == null ? "-" : `${(row.label_rate * 100).toFixed(1)}%`}</td>
                        <td>{row.lift == null ? "-" : row.lift.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="status">Load the review list first to see model numbers.</div>
          )}
        </section>
      ) : null}

      {tab === "queue" ? (
        <>
          <section className="metrics hero">
            <div className="metric">
              <span>Stays the tool checked</span>
              <strong>{data?.checked ?? "-"}</strong>
              <small>Discharged stays it looked at. Still in hospital do not appear here.</small>
            </div>
            <div className="metric">
              <span>On this list</span>
              <strong>{data?.top_reviews ?? "-"}</strong>
              <small>The 10 highest-priority stays for you to review.</small>
            </div>
          </section>

          <section className="layout">
            <div className="panel">
              <div className="panel-head">
                <h2>Review list</h2>
                <button type="button" onClick={() => loadReview()} disabled={loading}>
                  {loading ? "Checking stays..." : "Refresh list"}
                </button>
              </div>
              {!data && !loading ? (
                <div className="status">
                  Press Refresh list, or import records first. Highest-priority stays show at the top.
                </div>
              ) : null}
              {loading ? <div className="status">Checking stays. The first load can take a minute.</div> : null}
              <div className="list">
                {data?.reviews.map((row, index) => (
                  <div key={row.case_id} className={row.case_id === selectedId ? "row active" : "row"}>
                    <button className="row-main" type="button" onClick={() => setSelectedId(row.case_id)}>
                      <div className="row-top">
                        <span>
                          <span className="rank">#{index + 1}</span>{" "}
                          <span className="mono" title={row.case_id}>
                            Stay {shortId(row.case_id)}
                          </span>
                        </span>
                        <span className="who">
                          {row.age} / {row.gender}
                        </span>
                      </div>
                      <p className="why">{whyOnList(row)}</p>
                      <div className="chips">
                        {waitLabel(row) ? (
                          <span className={row.stale ? "chip stale" : "chip"}>{waitLabel(row)}</span>
                        ) : null}
                      </div>
                    </button>
                    {data.can_review ? (
                      <button
                        className="done-btn"
                        type="button"
                        onClick={() =>
                          markReviewed(row.case_id, "unclear", "cleared from queue").catch((err) =>
                            setError(err instanceof Error ? err.message : "Could not save review")
                          )
                        }
                      >
                        Done
                      </button>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>

            <div className="panel">
              <h2>This stay</h2>
              {!selected ? (
                <div className="status">Select a stay on the left.</div>
              ) : (
                <div className="detail">
                  <div>
                    <div className="row-top">
                      <strong className="mono" title={selected.case_id}>
                        Stay {shortId(selected.case_id)}
                      </strong>
                      <span className="who">
                        {selected.age}-year-old {selected.gender}
                      </span>
                    </div>
                    <p>
                      {selected.event_count} events on file.
                      {selected.waiting_days != null
                        ? ` Waiting ${selected.waiting_days} day${selected.waiting_days === 1 ? "" : "s"} since discharge.`
                        : ""}
                      {selected.stale ? " This one has been waiting a week or more." : ""}
                    </p>
                    {data?.can_review ? (
                      <div className="verdicts">
                        <button
                          type="button"
                          onClick={() =>
                            markReviewed(selected.case_id, "harm").catch((err) =>
                              setError(err instanceof Error ? err.message : "Could not save review")
                            )
                          }
                        >
                          Harm
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            markReviewed(selected.case_id, "no_harm").catch((err) =>
                              setError(err instanceof Error ? err.message : "Could not save review")
                            )
                          }
                        >
                          No harm
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            markReviewed(selected.case_id, "unclear").catch((err) =>
                              setError(err instanceof Error ? err.message : "Could not save review")
                            )
                          }
                        >
                          Unclear
                        </button>
                      </div>
                    ) : (
                      <p>Reviews save only for hospital stays in SafetyNet.</p>
                    )}
                  </div>
                  <div className="card">
                    <h3>Why it is on the list</h3>
                    {selected.triggers.length === 0 && selected.raising.length === 0 ? (
                      <div className="status tight">No specific flags were listed.</div>
                    ) : null}
                    {selected.triggers.map((text) => (
                      <div className="item" key={text}>
                        {plain(text)}
                      </div>
                    ))}
                    {selected.raising.slice(0, 4).map((item) => (
                      <div className="item" key={item.feature}>
                        {plain(item.description || item.feature)}
                      </div>
                    ))}
                  </div>
                  <div className="card">
                    <h3>What happened during the stay</h3>
                    {selected.timeline.length === 0 ? (
                      <div className="status tight">No events on file.</div>
                    ) : null}
                    {selected.timeline.map((event, idx) => (
                      <div className="item" key={`${event.timestamp}-${idx}`}>
                        <span className="mono">{event.timestamp}</span>
                        <br />
                        {event.event_type}: {event.value}
                        {event.details ? <em> - {event.details}</em> : null}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}
