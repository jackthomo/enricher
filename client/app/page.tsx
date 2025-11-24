"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Configurable = {
  model?: string;
  prompt?: string;
  max_search_results?: number;
  max_info_tool_calls?: number;
  max_loops?: number;
};

type EnrichmentResult = {
  info: Record<string, unknown>;
  trace?: TraceMessage[];
  steps?: StepEntry[];
};

type HealthStatus = "unknown" | "ok" | "error";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ||
  "http://localhost:8000";

type TraceMessage = {
  type: string;
  data?: {
    content?: unknown;
    name?: string;
    tool_call_id?: string;
    additional_kwargs?: Record<string, unknown>;
    tool_calls?: Array<{
      name?: string;
      args?: Record<string, unknown>;
    }>;
  };
};

type StepEntry = {
  event: string;
  node?: string;
  output_keys?: string[];
};

const DEFAULT_SCHEMA = JSON.stringify(
  {
    type: "object",
    properties: {
      providers: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string", description: "Company name" },
            technology_summary: {
              type: "string",
              description: "Brief summary of chip technology for LLM training",
            },
            current_market_share: {
              type: "string",
              description: "Estimated current market share",
            },
            future_outlook: {
              type: "string",
              description: "Outlook for the next 12 months",
            },
          },
          required: [
            "name",
            "technology_summary",
            "current_market_share",
            "future_outlook",
          ],
        },
        description: "List of top chip providers for LLM training",
      },
      overall_market_trends: {
        type: "string",
        description: "Brief paragraph on general trends in the LLM chip market",
      },
    },
    required: ["providers", "overall_market_trends"],
  },
  null,
  2
);

export default function Home() {
  const [topic, setTopic] = useState(
    "Top 5 chip providers for LLM training and their market outlook"
  );
  const [schemaText, setSchemaText] = useState(DEFAULT_SCHEMA);
  const [config, setConfig] = useState<Configurable>({
    model: "",
    prompt: "",
    max_search_results: undefined,
    max_info_tool_calls: undefined,
    max_loops: undefined,
  });
  const [result, setResult] = useState<EnrichmentResult | null>(null);
  const [trace, setTrace] = useState<TraceMessage[]>([]);
  const [steps, setSteps] = useState<StepEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [controller, setController] = useState<AbortController | null>(null);
  const [apiStatus, setApiStatus] = useState<HealthStatus>("unknown");
  const [apiStatusDetail, setApiStatusDetail] = useState("Not checked yet");
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const cleanedConfig = useMemo(() => {
    const entries = Object.entries(config).filter(
      ([, value]) =>
        value !== undefined &&
        value !== null &&
        value !== "" &&
        !Number.isNaN(value)
    );
    return entries.length ? Object.fromEntries(entries) : undefined;
  }, [config]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);
    setTrace([]);
    setSteps([]);
    const abortCtrl = new AbortController();
    setController(abortCtrl);
    let parsedSchema: Record<string, unknown>;
    try {
      parsedSchema = JSON.parse(schemaText);
    } catch (err) {
      setError("Extraction schema must be valid JSON");
      return;
    }

    if (apiStatus !== "ok") {
      const healthy = await checkHealth();
      if (!healthy) {
        setError("Backend health check failed. Please fix the connection.");
        return;
      }
    }

    const payload: {
      topic: string;
      extraction_schema: Record<string, unknown>;
      configurable?: Configurable;
    } = {
      topic,
      extraction_schema: parsedSchema,
    };
    if (cleanedConfig) {
      payload.configurable = cleanedConfig;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/enrich`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: abortCtrl.signal,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Request failed");
      }
      const data: EnrichmentResult = await response.json();
      console.info(
        "[enrich-ui] enrichment response",
        { traceLength: data.trace?.length ?? 0, stepCount: data.steps?.length ?? 0 }
      );
      setResult(data);
      setTrace(data.trace ?? []);
      setSteps(data.steps ?? []);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setError("Request cancelled");
        return;
      }
      const message =
        err instanceof Error ? err.message : "Something went wrong";
      setError(message);
    } finally {
      setIsLoading(false);
      setController(null);
    }
  };

  const checkHealth = async (): Promise<boolean> => {
    setIsCheckingHealth(true);
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as { status?: string };
      if (data?.status === "ok") {
        setApiStatus("ok");
        setApiStatusDetail("Backend reachable");
        return true;
      }
      throw new Error("Unexpected health response");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Health check failed";
      setApiStatus("error");
      setApiStatusDetail(message);
      return false;
    } finally {
      setIsCheckingHealth(false);
    }
  };

  useEffect(() => {
    void checkHealth();
  }, []);

  const handleCancel = () => {
    if (controller) {
      controller.abort();
      setController(null);
    }
    setIsLoading(false);
  };

  const infoDisplay = useMemo(() => {
    if (!result) return null;
    return JSON.stringify(result.info, null, 2);
  }, [result]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100 to-slate-200 text-slate-900">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <header className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-500">
            Data Enrichment
          </p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">
            Run the enrichment graph from the browser
          </h1>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-600">
            <span>
              API base:{" "}
              <code className="rounded bg-slate-900 px-2 py-1 text-xs text-slate-50">
                {API_BASE}
              </code>
            </span>
            <span
              className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
                apiStatus === "ok"
                  ? "bg-emerald-100 text-emerald-800"
                  : apiStatus === "error"
                    ? "bg-red-100 text-red-800"
                    : "bg-slate-200 text-slate-700"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  apiStatus === "ok"
                    ? "bg-emerald-500"
                    : apiStatus === "error"
                      ? "bg-red-500"
                      : "bg-slate-500"
                }`}
              />
              {apiStatus === "ok"
                ? "Backend healthy"
                : apiStatus === "error"
                  ? "Backend unreachable"
                  : "Health unknown"}
            </span>
            <button
              type="button"
              onClick={() => checkHealth()}
              disabled={isCheckingHealth}
              className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isCheckingHealth ? "Checking..." : "Check connection"}
            </button>
            <span className="text-xs text-slate-500">{apiStatusDetail}</span>
          </div>
        </header>

        <form
          onSubmit={handleSubmit}
          className="grid gap-6 lg:grid-cols-[1.4fr_1fr]"
        >
          <section className="space-y-4 rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-800">
                Topic
              </label>
              <input
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="What should we research?"
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-800">
                Extraction schema (JSON)
              </label>
              <textarea
                value={schemaText}
                onChange={(e) => setSchemaText(e.target.value)}
                rows={14}
                className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 font-mono text-xs text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
              />
              <p className="text-xs text-slate-500">
                Paste any JSON schema describing the structured output you need.
              </p>
            </div>

            <div className="grid gap-3 border-t border-slate-100 pt-4 sm:grid-cols-2">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-800">
                  Model (optional)
                </label>
                <input
                  value={config.model ?? ""}
                  onChange={(e) =>
                    setConfig((prev) => ({ ...prev, model: e.target.value }))
                  }
                  placeholder="anthropic/claude-3-5-sonnet-20240620"
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-800">
                  Max search results
                </label>
                <input
                  type="number"
                  min={1}
                  value={config.max_search_results ?? ""}
                  onChange={(e) =>
                    setConfig((prev) => ({
                      ...prev,
                      max_search_results:
                        e.target.value === ""
                          ? undefined
                          : Number(e.target.value),
                    }))
                  }
                  placeholder="5"
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-800">
                  Max Info tool calls
                </label>
                <input
                  type="number"
                  min={1}
                  value={config.max_info_tool_calls ?? ""}
                  onChange={(e) =>
                    setConfig((prev) => ({
                      ...prev,
                      max_info_tool_calls:
                        e.target.value === ""
                          ? undefined
                          : Number(e.target.value),
                    }))
                  }
                  placeholder="3"
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-800">
                  Max loops
                </label>
                <input
                  type="number"
                  min={1}
                  value={config.max_loops ?? ""}
                  onChange={(e) =>
                    setConfig((prev) => ({
                      ...prev,
                      max_loops:
                        e.target.value === ""
                          ? undefined
                          : Number(e.target.value),
                    }))
                  }
                  placeholder="6"
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-800">
                Prompt override (optional)
              </label>
              <textarea
                value={config.prompt ?? ""}
                onChange={(e) =>
                  setConfig((prev) => ({ ...prev, prompt: e.target.value }))
                }
                rows={4}
                placeholder="Custom prompt with {info} and {topic}"
                className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
              />
            </div>

            <div className="flex items-center justify-between pt-2">
              <p className="text-xs text-slate-500">
                Submit to FastAPI at {API_BASE}/enrich
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={!isLoading}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Pause/Cancel
                </button>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                >
                  {isLoading ? "Running..." : "Run enrichment"}
                </button>
              </div>
            </div>

            {error ? (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {error}
              </div>
            ) : null}
          </section>

          <section className="flex h-full flex-col gap-3 rounded-2xl bg-slate-900 p-6 text-slate-50 shadow-md">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
                  Result
                </p>
                <h2 className="text-lg font-semibold">Structured info</h2>
              </div>
              <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-100">
                Live
              </span>
            </div>
            <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
              <div className="min-h-[400px] rounded-xl bg-slate-950/50 p-4 font-mono text-xs text-slate-100 ring-1 ring-slate-700">
                {infoDisplay ? (
                  <pre className="whitespace-pre-wrap">{infoDisplay}</pre>
                ) : (
                  <p className="text-slate-400">
                    Submit a request to see the graph output here.
                  </p>
                )}
              </div>
              <div className="min-h-[400px] rounded-xl bg-slate-950/30 p-4 text-slate-100 ring-1 ring-slate-700">
                <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  <span>Trace</span>
                  <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] text-slate-200">
                    {trace.length} steps
                  </span>
                </div>
                <div className="space-y-2">
                  {trace.length === 0 ? (
                    <p className="text-slate-400">
                      Tool calls and agent reasoning will appear here.
                    </p>
                  ) : (
                    trace.map((msg, idx) => {
                      const toolCalls =
                        msg.data?.tool_calls ??
                        msg.data?.additional_kwargs?.tool_calls ??
                        [];
                      const content = msg.data?.content;
                      const contentText =
                        typeof content === "string"
                          ? content
                          : content
                            ? JSON.stringify(content, null, 2)
                            : "";
                      return (
                        <div
                          key={`${msg.type}-${idx}`}
                          className="rounded-lg bg-slate-900/60 p-3 ring-1 ring-slate-800"
                        >
                          <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.1em] text-slate-300">
                            <span>{msg.type}</span>
                            {msg.data?.name ? (
                              <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-200">
                                {msg.data.name}
                              </span>
                            ) : null}
                          </div>
                          {contentText ? (
                            <p className="text-xs text-slate-100">
                              {contentText}
                            </p>
                          ) : null}
                          {toolCalls.length ? (
                            <div className="mt-2 space-y-1 text-[11px] text-emerald-200">
                              {toolCalls.map((call, i) => (
                                <div
                                  key={`${call?.name ?? "tool"}-${i}`}
                                  className="rounded bg-emerald-900/30 px-2 py-1 ring-1 ring-emerald-800/60"
                                >
                                  <span className="font-semibold">
                                    {call?.name || "tool"}
                                  </span>
                                  {call?.args ? (
                                    <pre className="mt-1 whitespace-pre-wrap text-[10px] text-emerald-100">
                                      {JSON.stringify(call.args, null, 2)}
                                    </pre>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                          ) : null}
                          <details className="mt-2 text-[11px] text-slate-300">
                            <summary className="cursor-pointer text-slate-400">
                              Raw message
                            </summary>
                            <pre className="mt-1 whitespace-pre-wrap text-[10px] text-slate-200">
                              {JSON.stringify(msg, null, 2)}
                            </pre>
                          </details>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>
            <div className="rounded-2xl bg-slate-900/80 p-4 text-slate-100 ring-1 ring-slate-800">
              <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                <span>Node Steps</span>
                <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] text-slate-200">
                  {steps.length} events
                </span>
              </div>
              <div className="space-y-2">
                {steps.length === 0 ? (
                  <p className="text-slate-400">
                    Backend did not return node-level events.
                  </p>
                ) : (
                  steps.map((step, idx) => (
                    <div
                      key={`${step.node}-${idx}`}
                      className="rounded-lg bg-slate-800/60 p-3 ring-1 ring-slate-700"
                    >
                      <div className="flex items-center justify-between text-[11px] font-semibold text-slate-200">
                        <span>{step.node || "unknown node"}</span>
                        <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] text-slate-100">
                          {step.event.replace("on_", "")}
                        </span>
                      </div>
                      {step.output_keys && step.output_keys.length ? (
                        <p className="mt-1 text-[11px] text-slate-300">
                          Output keys: {step.output_keys.join(", ")}
                        </p>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>
        </form>
      </div>
    </div>
  );
}
