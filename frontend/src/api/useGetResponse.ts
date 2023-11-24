import { useAppStore } from '../store';
import { useMutation } from 'react-query';

const useGetResponse = () => {
  const setResponse = useAppStore((state) => state.setResponse);
  const request = useAppStore((state) => state.request);
  const confidence = useAppStore((state) => state.confidenceThreshold);

  const payload = {
    request,
    confidence,
  };

  const mutation = useMutation(
    async () => {
      const response = await fetch('your-api-endpoint', { body: JSON.stringify(payload) });
      const data = await response.json();
      return data;
    },
    {
      onSuccess: (data) => {
        setResponse(data);
      },
    },
  );

  return mutation;
};

export default useGetResponse;
