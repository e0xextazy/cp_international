import { Button, Fieldset } from '@mantine/core';
import styles from './ActionPanel.module.scss';
import { useAppStore } from '../../../../store';
import { ThresholdSlider } from '../../../../widgets/ThresholdSlider/ThresholdSlider';
import useGetResponse from '../../../../api/useGetResponse';

const BUTTON_PROPS = {
  size: 'md',
};

const Settings = () => {
  return (
    <div className={styles.settings}>
      <Fieldset legend="Confidence threshold" className={styles.settingsContent}>
        <ThresholdSlider />
      </Fieldset>
    </div>
  );
};

const Buttons = ({ isResponseForm, resetResponse, request, sendRequest, goToHome }) => {
  return (
    <div className={styles.buttons}>
      {isResponseForm ? (
        <Button {...BUTTON_PROPS} className={styles.sendBtn} onClick={resetResponse}>
          Отправить новый запрос
        </Button>
      ) : (
        <Button
          {...BUTTON_PROPS}
          disabled={!Boolean(request)}
          color="green"
          className={styles.sendBtn}
          onClick={sendRequest}
        >
          Отправить запрос
        </Button>
      )}
      <Button {...BUTTON_PROPS} variant="outline" onClick={goToHome}>
        На главную
      </Button>
    </div>
  );
};

export const ActionPanel = () => {
  const { setResponse, response, goToHome, request } = useAppStore();
  const { mutateAsync: fetchData } = useGetResponse();

  const sendRequest = async () => {
    try {
      await fetchData();
    } catch (error) {
      console.error(error);
    }
  };

  const resetResponse = () => {
    setResponse(null);
  };

  const isResponseForm = Boolean(response);

  return (
    <div className={styles.actionWrapper}>
      {!isResponseForm && <Settings />}
      <Buttons
        isResponseForm={isResponseForm}
        resetResponse={resetResponse}
        request={request}
        sendRequest={sendRequest}
        goToHome={goToHome}
      />
    </div>
  );
};
