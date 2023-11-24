import styles from './ActionPage.module.scss';

import { Response } from './sub/Response/Response';
import { SendRequestForm } from './sub/SendRequestForm/SendRequestForm';

import { useAppStore } from '../../store';
import { ActionPanel } from './sub/ActionPanel/ActionPanel';

export const ActionPage = () => {
  const { response } = useAppStore();
  const renderFormContent = () => {
    if (response) {
      return <Response className={styles.content} />;
    }

    return <SendRequestForm className={styles.content} />;
  };

  return (
    <div className={styles.page}>
      <div className={styles.form}>
        <div className={styles.contentWrapper}>{renderFormContent()}</div>
        <ActionPanel />
      </div>
    </div>
  );
};
