export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.BIOSEEK_API_BASE_URL ??
  "http://localhost:8000/api";

export type RetrievedDocument = {
  pmid: string;
  score: number;
  retrieval_text: string;
  metadata: {
    title: string;
    abstract: string;
    journal: string;
    year: number | null;
    mesh_terms: string[];
  };
  method: string;
  dataset_name: string;
};

export type SearchResponse = {
  query: string;
  method: string;
  dataset_name: string;
  documents: RetrievedDocument[];
  note: string;
};

export type CompareResponse = {
  query: string;
  dataset_name: string;
  results: {
    method: string;
    dataset_name: string;
    documents: RetrievedDocument[];
  }[];
  note: string;
};

export type ClusterResponse = {
  query: string;
  dataset_name: string;
  method: string;
  vector_space: string;
  num_clusters: number;
  status: string;
  error: string | null;
  silhouette_score: number | null;
  cluster_size_distribution: Record<string, number>;
  representative_term_summaries: Record<string, string[]>;
  cluster_summaries: {
    cluster_id: number;
    cluster_size: number;
    representative_keywords: string[];
    representative_docs: RetrievedDocument[];
  }[];
  diversified_results: RetrievedDocument[];
  artifact_paths: Record<string, string> | null;
};

export type SummaryResponse = {
  dataset_name: string;
  source_path: string;
  summary: Record<string, unknown>[] | Record<string, unknown>;
  narrative: Record<string, unknown> | null;
};

export type ArtifactResponse = {
  dataset_name: string;
  source_path: string;
  payload: Record<string, unknown>;
};

export type DatasetInfoResponse = {
  project_name: string;
  retrieval_corpus: string;
  relevance_labels: string;
  join_strategy: string;
  experimentation_scope: string;
  unified_dataset_note: string;
  stats: {
    pubmed_subset_docs: number | null;
    bioasq_questions: number | null;
    bioasq_unique_pmids: number | null;
    surviving_queries: number | null;
    mapped_relevant_docs: number | null;
    pmid_coverage_percent: number | null;
  };
  paths: Record<string, string>;
};

export type AIPolishResponse = {
  enabled: boolean;
  ai_assisted: boolean;
  mode: string;
  query: string;
  dataset_name: string | null;
  method: string | null;
  rewrite_suggestions: string[];
  suggested_follow_up_queries: string[];
  why_this_matched: {
    pmid: string;
    title: string;
    explanation: string;
  }[];
  refined_cluster_labels: {
    cluster_id: number;
    label: string;
    keywords: string[];
  }[];
  note: string;
};

export type QueryMetricsResponse = {
  dataset_name: string;
  source_path: string;
  query: string;
  found: boolean;
  payload: {
    message?: string;
    query_id?: string;
    query_text?: string;
    query_type?: string;
    num_relevant_docs?: number;
    rows?: Array<{
      method: string;
      query_id: string;
      query_text: string;
      query_type: string;
      num_relevant_docs: number;
      num_retrieved_docs: number;
      precision_at_5: number;
      precision_at_10: number;
      recall_at_10: number;
      mrr: number;
      ndcg_at_5: number;
      ndcg_at_10: number;
    }>;
  };
};

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // ignore JSON parse failures
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store"
  });
  return parseResponse<T>(response);
}

export async function apiPost<T>(path: string, body: object): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    cache: "no-store"
  });
  return parseResponse<T>(response);
}
