import { renderHook, act } from '@testing-library/react';
import { useAppStore } from '../index';
import { AppPages } from '../../app/config';

describe('useAppStore', () => {
  test('should set confidence threshold', () => {
    const { result } = renderHook(() => useAppStore());

    act(() => {
      result.current.setConfidenceThreshold(80);
    });

    expect(result.current.confidenceThreshold).toBe(80);
  });

  test('should set request', () => {
    const { result } = renderHook(() => useAppStore());

    act(() => {
      result.current.setRequest('sample request');
    });

    expect(result.current.request).toBe('sample request');
  });

  test('should set response', () => {
    const { result } = renderHook(() => useAppStore());

    act(() => {
      result.current.setResponse(null);
    });

    expect(result.current.response).toBe(null);
  });

  test('should set page', () => {
    const { result } = renderHook(() => useAppStore());

    act(() => {
      result.current.setPage(AppPages.ACTION_PAGE);
    });

    expect(result.current.page).toBe(AppPages.ACTION_PAGE);
  });

  test('should go to home', () => {
    const { result } = renderHook(() => useAppStore());

    act(() => {
      result.current.goToHome();
    });

    expect(result.current.page).toBe(AppPages.START_PAGE);
  });
});
