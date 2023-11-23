import { Button, Textarea, TextareaProps } from '@mantine/core';

import styles from './ActionPage.module.scss';

import { PageProps } from '../../shared/page';
import { AppPages } from '../../app/config';
import { useAutoAnimate } from '@formkit/auto-animate/react';
import { useState } from 'react';

const TEXTAREA_CONFIG: TextareaProps = {
  minRows: 6,
  maxRows: 12,
  autosize: true,
};

export const ActionPage = ({ onNav }: PageProps) => {
  const [animRef] = useAutoAnimate();
  const [showResult, setShowResult] = useState(false);

  const goToHome = () => {
    onNav(AppPages.START_PAGE);
  };

  const sendRequest = () => {
    setShowResult(true);
  };

  const reset = () => {
    setShowResult(false);
  };

  const renderFormContent = () => {
    if (showResult) {
      return (
        <>
          <h2 className={styles.title}>Результаты анализа обращения</h2>
          <img src="images/klim.jpg" height="400px" />
          <div className={styles.buttons}>
            <Button size="md" className={styles.sendBtn} onClick={reset}>
              Отправить новый запрос
            </Button>
            <Button size="md" variant="outline" onClick={goToHome}>
              На главную
            </Button>
          </div>
        </>
      );
    }

    return (
      <>
        <h2 className={styles.title}>Заполните обращение</h2>
        <Textarea variant="filled" label="Введите обращение гражданина" placeholder="Обращение" {...TEXTAREA_CONFIG} />
        <div className={styles.buttons}>
          <Button color="green" size="md" className={styles.sendBtn} onClick={sendRequest}>
            Отправить на обработку
          </Button>
          <Button size="md" variant="outline" onClick={goToHome}>
            На главную
          </Button>
        </div>
      </>
    );
  };

  return (
    <div className={styles.page}>
      <div className={styles.form} ref={animRef}>
        {renderFormContent()}
      </div>
    </div>
  );
};
