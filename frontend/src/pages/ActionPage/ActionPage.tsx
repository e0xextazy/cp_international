import styles from './ActionPage.module.scss';

import { useAutoAnimate } from '@formkit/auto-animate/react';
import { Response } from './sub/Response/Response';
import { SendRequestForm } from './sub/SendRequestForm/SendRequestForm';

import { useAppStore } from '../../store';
import { ActionPanel } from './sub/ActionPanel/ActionPanel';

export const ActionPage = () => {
  const { response } = useAppStore();

  const [ref] = useAutoAnimate();

  const renderFormContent = () => {
    if (response) {
      return <Response className={styles.content} />;
    }

    return <SendRequestForm className={styles.content} />;
  };

  return (
    <div className={styles.page}>
      <div className={styles.form} ref={ref}>
        {renderFormContent()}
        <ActionPanel />
      </div>
    </div>
  );
};
