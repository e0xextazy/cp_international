import { create } from 'zustand';
import { AppPages } from '../app/config';

export type ApiRequest = {
  appeal: string;
  confidenceThreshold: number;
};

export type TagsKeys = 'LOC' | 'ORG' | 'PER' | 'PHONE' | 'MONEY' | 'ADDRESS' | 'DATE';

export type ApiResponse = {
  executor: string | null;
  topic: string | null;
  subtopic: string | null;
  tags: Record<TagsKeys, string[]>;
};

export interface Store {
  page: AppPages;
  request: string | null;
  response: ApiResponse | null;
  confidenceThreshold: number;
  setConfidenceThreshold: (threshold: number) => void;
  setRequest: (request: string) => void;
  setResponse: (response: ApiResponse | null) => void;
  setPage: (page: AppPages) => void;
  goToHome: () => void;
}

const useAppStore = create<Store>((set) => ({
  request: null,
  response: null,
  confidenceThreshold: 95,
  page: AppPages.START_PAGE,
  setConfidenceThreshold: (threshold: number) => set(() => ({ confidenceThreshold: threshold })),
  setRequest: (request: string) => set(() => ({ request })),
  setResponse: (response: any) => set(() => ({ response })),
  setPage: (page: AppPages) => set(() => ({ page })),
  goToHome: () => set(() => ({ page: AppPages.START_PAGE })),
}));

export { useAppStore };
