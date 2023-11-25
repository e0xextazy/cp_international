import { ApiRequest, ApiResponse, useAppStore } from '../store';
import { useMutation } from 'react-query';

const API_URL = 'https://uni-ye-tile-spectrum.trycloudflare.com';

const useGetResponse = () => {
  const setResponse = useAppStore((state) => state.setResponse);
  const request = useAppStore((state) => state.request);
  const confidence = useAppStore((state) => state.confidenceThreshold);

  const payload: ApiRequest = {
    appeal: request as string,
    confidenceThreshold: confidence,
  };

  const mutation = useMutation(
    async () => {
      const response = await fetch(`${API_URL}/reco`, {
        body: JSON.stringify(payload),
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      return data;
    },
    {
      onSuccess: (data: ApiResponse) => {
        setResponse(data);
      },
      onError: (error) => {
        console.error(error);
      },
    },
  );

  return mutation;
};

export default useGetResponse;
