import { readFile } from "node:fs/promises";
import path from "node:path";

async function readMockJson<T>(fileName: string): Promise<T> {
  const target = path.join(process.cwd(), "public", "mock", fileName);
  const raw = await readFile(target, "utf8");
  return JSON.parse(raw) as T;
}

export async function loadProjectOverview() {
  return readMockJson<{
    dataset: {
      documents: number;
      queries: number;
      documentSource: string;
      querySource: string;
      joinKey: string;
    };
    analysis: {
      relevance: string;
      diversity: string;
      queryTypes: string;
    };
  }>("project-overview.json");
}

export async function loadSampleResults() {
  return readMockJson<
    {
      method: string;
      top_documents: {
        pmid: string;
        title: string;
        abstract_snippet: string;
        score: number;
        cluster: string;
        relevant: boolean;
        journal: string;
        year: number;
      }[];
    }[]
  >("sample-results.json");
}

export async function loadFrontendShell() {
  return readMockJson<{
    featuredQueries: string[];
    compare: {
      query: string;
      highlights: {
        label: string;
        value: string;
        note: string;
      }[];
      rows: {
        method: string;
        strength: string;
        tradeoff: string;
        bestFor: string;
      }[];
    };
    explore: {
      clusters: {
        clusterId: string;
        label: string;
        keywords: string[];
        size: number;
        documents: string[];
      }[];
      themeCoverage: {
        method: string;
        value: string;
      }[];
    };
    evaluation: {
      metrics: {
        method: string;
        p5: number;
        p10: number;
        recall10: number;
        ndcg5: number;
        ndcg10: number;
        mrr: number;
      }[];
      takeaways: {
        title: string;
        body: string;
      }[];
      queryBuckets: {
        bucket: string;
        winner: string;
        note: string;
      }[];
      examples: {
        method: string;
        bestQuery: string;
        bestNote: string;
        worstQuery: string;
        worstNote: string;
      }[];
      diversitySummary: {
        title: string;
        body: string;
      };
    };
    about: {
      principles: string[];
      pipeline: string[];
      datasetNotes: string[];
    };
  }>("frontend-shell.json");
}
