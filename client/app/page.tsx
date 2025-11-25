"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Configurable = {
  model?: string;
  prompt?: string;
  max_search_results?: number;
  max_info_tool_calls?: number;
  max_loops?: number;
  max_time_seconds?: number;
};

type EnrichmentResult = {
  info: Record<string, unknown>;
  trace?: TraceMessage[];
  steps?: StepEntry[];
  metrics?: {
    duration_ms?: number;
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
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

type JsonSchema = {
  type?: string | string[];
  description?: string;
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
};

type SchemaFieldRow = {
  field: string;
  type: string;
  required: boolean;
  description?: string;
};

type ColumnType = "string" | "number" | "boolean" | "array" | "object";

type ColumnSpec = {
  id: string;
  name: string;
  type: ColumnType;
  description: string;
  required: boolean;
};

type CsvHeader = {
  key: string;
  label: string;
};

type ParsedCsv = {
  headers: CsvHeader[];
  rows: Record<string, string>[];
};

const toTableRows = (data: unknown): Array<{ key: string; value: string }> => {
  if (data === null || data === undefined) {
    return [{ key: "value", value: "null" }];
  }
  if (typeof data !== "object") {
    return [{ key: "value", value: String(data) }];
  }
  if (Array.isArray(data)) {
    return data.map((item, idx) => ({
      key: `[${idx}]`,
      value:
        typeof item === "object" ? JSON.stringify(item, null, 2) : String(item),
    }));
  }
  return Object.entries(data).map(([k, v]) => ({
    key: k,
    value: typeof v === "object" ? JSON.stringify(v, null, 2) : String(v),
  }));
};

const isArrayOfObjects = (val: unknown): val is Array<Record<string, unknown>> =>
  Array.isArray(val) && val.length > 0 && val.every((item) => typeof item === "object" && item !== null);

const makeId = () => Math.random().toString(36).slice(2, 9);

const createColumn = (
  name: string,
  description: string,
  options?: { required?: boolean; type?: ColumnType }
): ColumnSpec => ({
  id: `${name || "column"}-${makeId()}`,
  name,
  description,
  required: options?.required ?? true,
  type: options?.type ?? "string",
});

const DEFAULT_COLUMNS: ColumnSpec[] = [
  createColumn("name", "Company name"),
  createColumn("technology_summary", "Brief summary of chip technology for LLM training"),
  createColumn("current_market_share", "Estimated current market share"),
  createColumn("future_outlook", "Outlook for the next 12 months"),
];

const parseCsvLine = (line: string): string[] => {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"' && line[i - 1] !== "\\") {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current);
  return values.map((v) => v.trim());
};

const normalizeHeaderKey = (label: string, index: number) => {
  const cleaned = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || `column_${index + 1}`;
};

const parseCsvText = (text: string): ParsedCsv => {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) {
    throw new Error("CSV file is empty.");
  }

  const headerCells = parseCsvLine(lines[0]);
  if (!headerCells.length) {
    throw new Error("No header row detected.");
  }

  const seenKeys = new Set<string>();
  const headers: CsvHeader[] = headerCells.map((label, idx) => {
    let key = normalizeHeaderKey(label, idx);
    while (seenKeys.has(key)) {
      key = `${key}_${idx + 1}`;
    }
    seenKeys.add(key);
    return {
      key,
      label: label || `column_${idx + 1}`,
    };
  });

  const rows = lines
    .slice(1)
    .map((line) => {
      const cells = parseCsvLine(line);
      const row: Record<string, string> = {};
      headers.forEach((header, idx) => {
        row[header.key] = cells[idx]?.trim() ?? "";
      });
      return row;
    })
    .filter((row) => Object.values(row).some((value) => value.trim().length > 0));

  return { headers, rows };
};

const inferColumnTypeFromValues = (values: string[]): ColumnType => {
  const nonEmpty = values.filter((v) => v.trim().length > 0);
  if (!nonEmpty.length) return "string";

  const allNumbers = nonEmpty.every((v) => !Number.isNaN(Number(v)));
  if (allNumbers) return "number";

  const lower = nonEmpty.map((v) => v.toLowerCase());
  const allBooleans = lower.every((v) => v === "true" || v === "false" || v === "yes" || v === "no");
  if (allBooleans) return "boolean";

  const allArrays = nonEmpty.every((v) => v.startsWith("[") && v.endsWith("]"));
  if (allArrays) return "array";
  const allObjects = nonEmpty.every((v) => v.startsWith("{") && v.endsWith("}"));
  if (allObjects) return "object";

  return "string";
};

const buildColumnsFromCsv = (parsed: ParsedCsv): ColumnSpec[] => {
  return parsed.headers.map((header, idx) => {
    const values = parsed.rows.map((row) => row[header.key] ?? "");
    return createColumn(
      header.key,
      `Inferred from CSV header "${header.label}"`,
      {
        required: values.some((value) => value.trim().length > 0),
        type: inferColumnTypeFromValues(values),
      }
    );
  });
};

const normalizeSchemaType = (node: JsonSchema): string => {
  if (Array.isArray(node.type)) return node.type.join(" | ");
  if (node.type) return node.type;
  if (node.properties) return "object";
  if (node.items) return "array";
  return "unknown";
};

const extractSchemaFields = (
  schema: unknown,
  path = "",
  isRequired = false
): SchemaFieldRow[] => {
  if (!schema || typeof schema !== "object") return [];
  const node = schema as JsonSchema;
  const nodeType = normalizeSchemaType(node);
  const description = typeof node.description === "string" ? node.description : "";

  if (nodeType === "object" && node.properties) {
    const requiredSet = new Set(node.required ?? []);
    return Object.entries(node.properties).flatMap(([key, value]) => {
      const childPath = path ? `${path}.${key}` : key;
      const childRequired = requiredSet.has(key);
      return extractSchemaFields(value, childPath, childRequired);
    });
  }

  if (nodeType === "array" && node.items) {
    const itemPath = `${path}[]`;
    return extractSchemaFields(node.items, itemPath, isRequired);
  }

  if (!path) return [];

  return [
    {
      field: path,
      type: nodeType,
      required: isRequired,
      description,
    },
  ];
};

const InfoTableView = ({ info }: { info: Record<string, unknown> }) => {
  const entries = Object.entries(info);
  const simpleEntries = entries.filter(
    ([, value]) => value === null || value === undefined || typeof value !== "object"
  );
  const complexEntries = entries.filter(([, value]) => !(value === null || value === undefined || typeof value !== "object"));

  return (
    <div className="space-y-4">
      {simpleEntries.length ? (
        <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-950/40">
          <table className="min-w-full text-left text-xs text-slate-100">
            <thead className="bg-slate-900/70 text-[11px] uppercase tracking-[0.1em] text-slate-400">
              <tr>
                {simpleEntries.map(([key]) => (
                  <th key={key} className="px-3 py-2">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="bg-slate-900/30">
                {simpleEntries.map(([key, value]) => (
                  <td key={key} className="px-3 py-2 font-mono whitespace-pre-wrap text-slate-100">
                    {String(value ?? "")}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      ) : null}

      {complexEntries.map(([key, value]) => {
        if (isArrayOfObjects(value)) {
          const columns = Array.from(
            value.reduce(
              (set, row) => {
                Object.keys(row).forEach((k) => set.add(k));
                return set;
              },
              new Set<string>()
            )
          );
          return (
            <div key={key}>
              <div className="mb-1 text-sm font-semibold text-slate-100">
                {key}
              </div>
              <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-950/40">
                <table className="min-w-full text-left text-xs text-slate-100">
                  <thead className="bg-slate-900/70 text-[11px] uppercase tracking-[0.1em] text-slate-400">
                    <tr>
                      {columns.map((col) => (
                        <th key={col} className="px-3 py-2">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {value.map((row, ridx) => (
                      <tr
                        key={`${key}-row-${ridx}`}
                        className={
                          ridx % 2 === 0 ? "bg-slate-900/40" : "bg-slate-900/20"
                        }
                      >
                        {columns.map((col) => (
                          <td key={`${col}-${ridx}`} className="px-3 py-2">
                            <span className="font-mono whitespace-pre-wrap text-slate-100">
                              {typeof row[col] === "object"
                                ? JSON.stringify(row[col], null, 2)
                                : String(row[col] ?? "")}
                            </span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        }
        const rows = toTableRows(value);
        return (
          <div key={key}>
            <div className="mb-1 text-sm font-semibold text-slate-100">
              {key}
            </div>
            <div className="overflow-x-auto rounded-md border border-slate-800 bg-slate-950/40">
              <table className="min-w-full text-left text-xs text-slate-100">
                <thead className="bg-slate-900/70 text-[11px] uppercase tracking-[0.1em] text-slate-400">
                  <tr>
                    <th className="px-3 py-2">Key</th>
                    <th className="px-3 py-2">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, idx) => (
                    <tr
                      key={`${row.key}-${idx}`}
                      className={
                        idx % 2 === 0 ? "bg-slate-900/40" : "bg-slate-900/20"
                      }
                    >
                      <td className="px-3 py-2 font-semibold">{row.key}</td>
                      <td className="px-3 py-2 font-mono whitespace-pre-wrap text-slate-100">
                        {row.value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default function Home() {
  const [topic, setTopic] = useState(
    "Top 5 chip providers for LLM training and their market outlook"
  );
  const [columns, setColumns] = useState<ColumnSpec[]>(DEFAULT_COLUMNS);
  const [inputRows, setInputRows] = useState<Record<string, string>[]>([]);
  const [csvMeta, setCsvMeta] = useState<{
    fileName: string;
    rowCount: number;
    columns: string[];
  } | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [targetRowCount, setTargetRowCount] = useState<number | "">(5);
  const [config, setConfig] = useState<Configurable>({
    model: "",
    prompt: "",
    max_search_results: undefined,
    max_info_tool_calls: undefined,
    max_loops: undefined,
    max_time_seconds: undefined,
  });
  const [result, setResult] = useState<EnrichmentResult | null>(null);
  const [trace, setTrace] = useState<TraceMessage[]>([]);
  const [steps, setSteps] = useState<StepEntry[]>([]);
  const [metrics, setMetrics] = useState<EnrichmentResult["metrics"]>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [controller, setController] = useState<AbortController | null>(null);
  const [apiStatus, setApiStatus] = useState<HealthStatus>("unknown");
  const [apiStatusDetail, setApiStatusDetail] = useState("Not checked yet");
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [resultView, setResultView] = useState<"table" | "json">("table");
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showTrace, setShowTrace] = useState(true);

  const csvPreviewHeaders = useMemo(() => {
    const headerSet = new Set<string>();
    inputRows.forEach((row) => Object.keys(row).forEach((key) => headerSet.add(key)));
    return Array.from(headerSet);
  }, [inputRows]);

  const csvPreviewRows = useMemo(() => inputRows.slice(0, 5), [inputRows]);

  const handleColumnChange = (
    id: string,
    key: "name" | "type" | "description" | "required",
    value: string | boolean
  ) => {
    if (key === "name" && typeof value === "string" && inputRows.length) {
      const current = columns.find((col) => col.id === id);
      if (current?.name && current.name !== value) {
        setInputRows((prevRows) =>
          prevRows.map((row) => {
            if (!(current.name in row)) return row;
            const { [current.name]: existing, ...rest } = row;
            return { ...rest, [value]: existing };
          })
        );
      }
    }
    setColumns((prev) =>
      prev.map((col) => (col.id === id ? { ...col, [key]: value } : col))
    );
  };

  const handleRemoveColumn = (id: string) => {
    setColumns((prev) => (prev.length <= 1 ? prev : prev.filter((col) => col.id !== id)));
  };

  const handleAddColumn = () => {
    setColumns((prev) => [
      ...prev,
      createColumn(
        `column_${prev.length + 1}`,
        "Describe this field for the model",
        { required: false }
      ),
    ]);
  };

  const applyCsvToState = (parsed: ParsedCsv, fileName: string) => {
    const inferredColumns = buildColumnsFromCsv(parsed);
    setColumns(inferredColumns);
    setInputRows(parsed.rows);
    setCsvMeta({
      fileName,
      rowCount: parsed.rows.length,
      columns: inferredColumns.map((col) => col.name || col.id),
    });
    setTargetRowCount((prev) => (parsed.rows.length > 0 ? parsed.rows.length : prev));
    setCsvError(null);
  };

  const handleCsvFile = (files: FileList | null) => {
    if (!files?.length) return;
    const file = files[0];
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = String(event.target?.result || "");
        const parsed = parseCsvText(text);
        applyCsvToState(parsed, file.name);
      } catch (err) {
        setCsvError(err instanceof Error ? err.message : "Failed to parse CSV");
        setCsvMeta(null);
        setInputRows([]);
      }
    };
    reader.readAsText(file);
  };

  const handleClearCsv = () => {
    setCsvMeta(null);
    setInputRows([]);
    setCsvError(null);
  };

  const handleRowCountChange = (value: string) => {
    if (value === "") {
      setTargetRowCount("");
      return;
    }
    const asNumber = Number(value);
    if (Number.isNaN(asNumber) || asNumber < 1) return;
    setTargetRowCount(asNumber);
  };

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
    setMetrics(undefined);
    const abortCtrl = new AbortController();
    setController(abortCtrl);
    const hasColumns = columns.some((col) => col.name.trim().length > 0);
    if (!hasColumns) {
      setError("Add at least one column name to build the dataset schema.");
      return;
    }

    if (apiStatus !== "ok") {
      const healthy = await checkHealth();
      if (!healthy) {
        setError("Backend health check failed. Please fix the connection.");
        return;
      }
    }

    const topicNotes: string[] = [topic.trim()];
    if (typeof targetRowCount === "number" && targetRowCount > 0) {
      topicNotes.push(
        `Target row count: ${targetRowCount}. Keep the providers array aligned to this batch size.`
      );
    }
    const topicForRequest = topicNotes.filter(Boolean).join("\n\n");

    const payload: {
      topic: string;
      extraction_schema: Record<string, unknown>;
      configurable?: Configurable;
      input_rows?: Record<string, string>[];
    } = {
      topic: topicForRequest,
      extraction_schema: extractionSchema,
    };
    if (cleanedConfig) {
      payload.configurable = cleanedConfig;
    }
    if (inputRows.length) {
      payload.input_rows = inputRows;
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
      setMetrics(data.metrics);
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

  const extractionSchema = useMemo(() => {
    const properties: Record<string, JsonSchema> = {};
    const required: string[] = [];
    columns.forEach((col, idx) => {
      const key = col.name.trim() || `column_${idx + 1}`;
      properties[key] = {
        type: col.type,
        description: col.description || undefined,
      };
      if (col.required) {
        required.push(key);
      }
    });
    const schema: { type: string; properties: Record<string, JsonSchema>; required: string[] } = {
      type: "object",
      properties: {
        providers: {
          type: "array",
          items: {
            type: "object",
            properties,
            required,
          },
          description: [
            "Dataset rows for enrichment.",
            inputRows.length
              ? `User supplied ${inputRows.length} row(s) from CSV. Preserve provided values and fill missing fields for these rows. Do not add extra rows unless instructed.`
              : typeof targetRowCount === "number" && targetRowCount > 0
                ? `Return about ${targetRowCount} row(s) in this array.`
                : null,
          ]
            .filter(Boolean)
            .join(" "),
        },
        summary: {
          type: "string",
          description: "Add a crisp 2-3 sentence summary capturing key trends across providers.",
        },
      },
      required: ["providers", "summary"],
    };
    return schema;
  }, [columns, inputRows.length, targetRowCount]);

  const schemaText = useMemo(
    () => JSON.stringify(extractionSchema, null, 2),
    [extractionSchema]
  );

  const infoDisplay = useMemo(() => {
    if (!result) return null;
    return JSON.stringify(result.info, null, 2);
  }, [result]);

  const schemaFieldRows = useMemo(
    () => extractSchemaFields(extractionSchema),
    [extractionSchema]
  );

  useEffect(() => {
    if (result) {
      setResultView("table");
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100 to-slate-200 text-slate-900">
      <div className="mx-auto w-full max-w-7xl px-6 py-10 lg:px-10">
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

        <form onSubmit={handleSubmit} className="space-y-6">
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

            <div className="space-y-3 rounded-2xl border border-slate-100 bg-slate-50 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Input batch (CSV optional)
                  </p>
                  <p className="text-xs text-slate-500">
                    Upload a CSV with partially filled rows to infer columns and batch size. We send these rows to the model so it can fill any gaps.
                  </p>
                </div>
                {csvMeta ? (
                  <button
                    type="button"
                    onClick={handleClearCsv}
                    className="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-slate-50"
                  >
                    Clear CSV
                  </button>
                ) : null}
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <label className="flex cursor-pointer flex-col gap-2 rounded-xl border border-dashed border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-800 transition hover:border-slate-400 hover:bg-slate-50">
                  <input
                    type="file"
                    accept=".csv,text/csv"
                    className="hidden"
                    onChange={(e) => handleCsvFile(e.target.files)}
                  />
                  <span>{csvMeta ? `Loaded ${csvMeta.fileName}` : "Upload CSV"}</span>
                  <span className="text-xs font-normal text-slate-500">
                    {csvMeta
                      ? `Detected ${csvMeta.rowCount} row(s) and ${csvMeta.columns.length} column(s). Columns below update automatically.`
                      : "We infer column names, types, and batch size from your file."}
                  </span>
                </label>
                <div className="space-y-1">
                  <label className="text-[11px] uppercase tracking-[0.1em] text-slate-600">
                    Target rows
                  </label>
                  <input
                    type="number"
                    min={1}
                    value={targetRowCount}
                    onChange={(e) => handleRowCountChange(e.target.value)}
                    className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                  />
                  <p className="text-xs text-slate-500">
                    Auto-filled from CSV. Otherwise use this to request a specific batch size (e.g. Top 5).
                  </p>
                </div>
              </div>
              {csvError ? (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {csvError}
                </div>
              ) : null}
              {inputRows.length ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Previewing first {Math.min(5, inputRows.length)} of {inputRows.length} row(s)
                    </span>
                    <span className="rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-semibold text-emerald-700">
                      Attached to request for gap-filling
                    </span>
                  </div>
                  <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
                    <table className="min-w-full text-left text-xs text-slate-800">
                      <thead className="bg-slate-100 text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        <tr>
                          {csvPreviewHeaders.map((header) => (
                            <th key={header} className="px-3 py-2">
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {csvPreviewRows.map((row, idx) => (
                          <tr
                            key={`csv-row-${idx}`}
                            className="odd:bg-slate-50 even:bg-white"
                          >
                            {csvPreviewHeaders.map((header) => (
                              <td
                                key={`${header}-${idx}`}
                                className="px-3 py-2 font-mono text-[11px] text-slate-700"
                              >
                                {row[header] || "—"}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-800">
                Dataset columns
              </label>
              <p className="text-xs text-slate-500">
                Upload a CSV to auto-infer columns, or build them directly. The JSON schema updates automatically.
              </p>
              <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
                <table className="min-w-[820px] text-left text-sm text-slate-800">
                  <thead className="bg-slate-100 text-[11px] uppercase tracking-[0.1em] text-slate-500">
                    <tr>
                      <th className="px-3 py-2 w-[140px]">Field</th>
                      {columns.map((col) => (
                        <th key={col.id} className="px-3 py-2">
                          <input
                            value={col.name}
                            onChange={(e) => handleColumnChange(col.id, "name", e.target.value)}
                            placeholder="column_name"
                            className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-900 outline-none transition focus:border-slate-400"
                          />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="odd:bg-slate-50 even:bg-white">
                      <td className="px-3 py-2 font-semibold text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        Type
                      </td>
                      {columns.map((col) => (
                        <td key={`${col.id}-type`} className="px-3 py-2 align-top">
                          <select
                            value={col.type}
                            onChange={(e) =>
                              handleColumnChange(col.id, "type", e.target.value as ColumnType)
                            }
                            className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                          >
                            <option value="string">string</option>
                            <option value="number">number</option>
                            <option value="boolean">boolean</option>
                            <option value="array">array</option>
                            <option value="object">object</option>
                          </select>
                        </td>
                      ))}
                    </tr>
                    <tr className="odd:bg-slate-50 even:bg-white">
                      <td className="px-3 py-2 font-semibold text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        Required
                      </td>
                      {columns.map((col) => (
                        <td key={`${col.id}-required`} className="px-3 py-2 align-top">
                          <button
                            type="button"
                            onClick={() => handleColumnChange(col.id, "required", !col.required)}
                            className={`w-full rounded-full px-3 py-2 text-[12px] font-semibold transition ${
                              col.required
                                ? "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200"
                                : "bg-slate-200 text-slate-700 ring-1 ring-slate-300 hover:bg-slate-300"
                            }`}
                          >
                            {col.required ? "Required" : "Optional"}
                          </button>
                        </td>
                      ))}
                    </tr>
                    <tr className="odd:bg-slate-50 even:bg-white">
                      <td className="px-3 py-2 font-semibold text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        Description
                      </td>
                      {columns.map((col) => (
                        <td key={`${col.id}-description`} className="px-3 py-2 align-top">
                          <textarea
                            value={col.description}
                            onChange={(e) => handleColumnChange(col.id, "description", e.target.value)}
                            rows={2}
                            placeholder="What should the model provide?"
                            className="w-full resize-none rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                          />
                        </td>
                      ))}
                    </tr>
                    <tr className="odd:bg-slate-50 even:bg-white">
                      <td className="px-3 py-2 font-semibold text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        Actions
                      </td>
                      {columns.map((col) => (
                        <td key={`${col.id}-actions`} className="px-3 py-2 align-top">
                          <button
                            type="button"
                            onClick={() => handleRemoveColumn(col.id)}
                            disabled={columns.length <= 1}
                            className="w-full rounded-full border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:border-slate-400 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            Remove
                          </button>
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={handleAddColumn}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition hover:border-slate-400 hover:bg-slate-50"
                >
                  + Add column
                </button>
                <p className="text-xs text-slate-500">
                  Tables scroll horizontally on smaller screens.
                </p>
              </div>
            </div>
            <details className="rounded-lg border border-slate-100 bg-slate-50 p-3">
              <summary className="flex cursor-pointer items-center justify-between text-sm font-semibold text-slate-800">
                <span>Generated schema (optional view)</span>
                <span className="text-[11px] text-slate-500">Click to expand</span>
              </summary>
              <div className="mt-3 space-y-3">
                <div className="max-h-[280px] overflow-auto rounded-md border border-slate-200 bg-white">
                  <pre className="whitespace-pre px-3 py-2 font-mono text-xs text-slate-800">
                    {schemaText}
                  </pre>
                </div>
                {schemaFieldRows.length ? (
                  <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
                    <table className="min-w-full text-left text-xs text-slate-800">
                      <thead className="bg-slate-100 text-[11px] uppercase tracking-[0.1em] text-slate-500">
                        <tr>
                          <th className="px-3 py-2">Field</th>
                          <th className="px-3 py-2">Type</th>
                          <th className="px-3 py-2">Required</th>
                          <th className="px-3 py-2">Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {schemaFieldRows.map((row) => (
                          <tr
                            key={row.field}
                            className="odd:bg-slate-50 even:bg-white"
                          >
                            <td className="px-3 py-2 font-semibold font-mono">
                              {row.field}
                            </td>
                            <td className="px-3 py-2">{row.type}</td>
                            <td className="px-3 py-2">
                              <span
                                className={`inline-flex rounded-full px-2 py-1 text-[11px] font-semibold ${
                                  row.required
                                    ? "bg-emerald-100 text-emerald-700"
                                    : "bg-slate-200 text-slate-700"
                                }`}
                              >
                                {row.required ? "Yes" : "No"}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-slate-600">
                              {row.description || "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    Add columns to generate a schema preview.
                  </p>
                )}
              </div>
            </details>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Run settings
                </p>
                <p className="text-xs text-slate-500">
                  Model, limits, and prompt overrides live in the settings modal.
                </p>
                <p className="text-xs text-slate-500">
                  Current model: {config.model ? config.model : "Provider default"}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowConfigModal(true)}
                className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition hover:border-slate-400 hover:bg-slate-50"
              >
                Open settings
              </button>
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
              {metrics ? (
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  {metrics.duration_ms !== undefined ? (
                    <span className="rounded-full bg-slate-800 px-2 py-1 text-slate-100">
                      Time: {(metrics.duration_ms / 1000).toFixed(1)}s
                    </span>
                  ) : null}
                  {metrics.total_tokens !== undefined ? (
                    <span className="rounded-full bg-slate-800 px-2 py-1 text-slate-100">
                      Tokens: {metrics.total_tokens} (in {metrics.input_tokens ?? "?"} / out {metrics.output_tokens ?? "?"})
                    </span>
                  ) : null}
                </div>
              ) : null}
              <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-semibold text-emerald-100">
                Live
              </span>
            </div>
            <div className="min-h-[320px] rounded-xl bg-slate-950/50 p-4 text-slate-100 ring-1 ring-slate-700">
              <div className="mb-3 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setResultView("table")}
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    resultView === "table"
                      ? "bg-white text-slate-900"
                      : "bg-slate-800 text-slate-200 hover:bg-slate-700"
                  }`}
                >
                  Table view
                </button>
                <button
                  type="button"
                  onClick={() => setResultView("json")}
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition ${
                    resultView === "json"
                      ? "bg-white text-slate-900"
                      : "bg-slate-800 text-slate-200 hover:bg-slate-700"
                  }`}
                >
                  Raw JSON
                </button>
                <span className="text-[11px] text-slate-400">
                  Toggle how enriched data is shown
                </span>
              </div>
              <div className="overflow-x-auto">
                {result ? (
                  resultView === "table" ? (
                    <InfoTableView info={result.info} />
                  ) : infoDisplay ? (
                    <pre className="font-mono text-xs whitespace-pre-wrap">
                      {infoDisplay}
                    </pre>
                  ) : null
                ) : (
                  <p className="text-slate-400">
                    Submit a request to see the graph output here.
                  </p>
                )}
              </div>
            </div>
            <div className="rounded-xl bg-slate-950/30 p-4 text-slate-100 ring-1 ring-slate-700">
              <details open={showTrace} className="group">
                <summary
                  onClick={() => setShowTrace((prev) => !prev)}
                  className="flex cursor-pointer items-center justify-between text-xs font-semibold uppercase tracking-[0.2em] text-slate-400"
                >
                  <span>Trace</span>
                  <span className="rounded-full bg-slate-800 px-2 py-1 text-[10px] text-slate-200">
                    {trace.length} steps
                  </span>
                </summary>
                <div className="mt-2 space-y-2">
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
              </details>
            </div>
          </section>
        </form>
        {showConfigModal ? (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4">
            <div
              className="absolute inset-0"
              onClick={() => setShowConfigModal(false)}
              aria-label="Close settings"
            />
            <div className="relative z-10 w-full max-w-3xl rounded-2xl bg-white p-6 shadow-2xl ring-1 ring-slate-200">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                    Run settings
                  </p>
                  <h3 className="text-lg font-semibold text-slate-900">
                    Model & limits
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={() => setShowConfigModal(false)}
                  className="rounded-full border border-slate-200 px-3 py-1 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
                >
                  Close
                </button>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
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
                <div className="space-y-2">
                  <label className="text-sm font-semibold text-slate-800">
                    Max time (seconds)
                  </label>
                  <input
                    type="number"
                    min={10}
                    value={config.max_time_seconds ?? ""}
                    onChange={(e) =>
                      setConfig((prev) => ({
                        ...prev,
                        max_time_seconds:
                          e.target.value === ""
                            ? undefined
                            : Number(e.target.value),
                      }))
                    }
                    placeholder="120"
                    className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                  />
                </div>
              </div>
              <div className="mt-4 space-y-2">
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
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
