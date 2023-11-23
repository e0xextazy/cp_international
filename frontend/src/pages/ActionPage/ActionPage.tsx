import { Button, Textarea, TextareaProps } from '@mantine/core';

import styles from './ActionPage.module.scss';

import { PageProps } from '../../shared/page';
import { AppPages } from '../../app/config';

const TEXTAREA_CONFIG: TextareaProps = {
  minRows: 6,
  maxRows: 12,
  autosize: true,
};

export const ActionPage = ({ onNav }: PageProps) => {
  const goToHome = () => {
    onNav(AppPages.START_PAGE);
  };

  return (
    <div className={styles.page}>
      <div className={styles.form}>
        <h2 className={styles.title}>Заполните обращение</h2>
        <Textarea variant="filled" label="Введите обращение гражданина" placeholder="Обращение" {...TEXTAREA_CONFIG} />
        <div className={styles.buttons}>
          <Button size="md" className={styles.sendBtn} loading>
            Отправить на обработку
          </Button>
          <Button size="md" variant="outline" onClick={goToHome}>
            На главную
          </Button>
        </div>
      </div>
    </div>
  );
};
