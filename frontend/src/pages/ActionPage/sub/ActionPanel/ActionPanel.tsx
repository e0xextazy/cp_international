import { Button } from '@mantine/core';
import styles from './ActionPanel.module.scss';
import { useAppStore } from '../../../../store';

export const ActionPanel = () => {
  const { setResponse, response, goToHome } = useAppStore();

  const resetResponse = () => {
    setResponse(null);
  };

  const sendResponse = () => {
    setResponse(' ');
  };

  const isResponseForm = Boolean(response);

  return (
    <div className={styles.buttons}>
      {isResponseForm ? (
        <Button size="md" className={styles.sendBtn} onClick={resetResponse}>
          Отправить новый запрос
        </Button>
      ) : (
        <Button size="md" color="green" className={styles.sendBtn} onClick={sendResponse}>
          Отправить запрос
        </Button>
      )}
      <Button size="md" variant="outline" onClick={goToHome}>
        На главную
      </Button>
    </div>
  );
};
