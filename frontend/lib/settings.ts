import { readFile } from "node:fs/promises";
import path from "node:path";

type RawSettings = {
  project: {
    name: string;
    tagline: string;
    research_question: string;
  };
  retrieval_methods: {
    id: string;
    label: string;
    family: string;
    description: string;
  }[];
  analysis_modules: string[];
  frontend: {
    default_query: string;
    results_per_method: number;
  };
};

export async function loadSharedSettings() {
  const filePath = path.join(process.cwd(), "..", "shared", "config", "settings.json");
  const raw = await readFile(filePath, "utf8");
  const typed = JSON.parse(raw) as RawSettings;

  return {
    project: {
      name: typed.project.name,
      tagline: typed.project.tagline,
      researchQuestion: typed.project.research_question
    },
    retrievalMethods: typed.retrieval_methods,
    analysisModules: typed.analysis_modules,
    frontend: typed.frontend
  };
}
