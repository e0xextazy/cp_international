import { create } from 'zustand';
import { AppPages } from '../app/config';

export interface Store {
  page: AppPages;
  request: string | null;
  response: any | null;
  confidenceThreshold: number;
  setConfidenceThreshold: (threshold: number) => void;
  setRequest: (request: string) => void;
  setResponse: (response: any) => void;
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
