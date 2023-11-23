import { useGlitch } from 'react-powerglitch';
import { IconArrowRight } from '@tabler/icons-react';
import { Button } from '@mantine/core';

import styles from './StartScreenPage.module.scss';
import { AppPages } from '../../app/config';
import { useAppStore } from '../../store';

export const StartScreenPage = () => {
  const { setPage } = useAppStore();
  const glitch = useGlitch();

  const goToAction = () => {
    setPage(AppPages.ACTION_PAGE);
  };

  return (
    <div className={styles.page}>
      <div className={styles.banner}>
        <div className={styles.logo}>
          <h1>AppealClassifier</h1>
        </div>
        <span ref={glitch.ref}>Powerful AI powered tool</span>
      </div>
      <div className={styles.action}>
        <Button
          onClick={goToAction}
          fullWidth
          variant="light"
          justify="space-between"
          rightSection={<IconArrowRight />}
          className={styles.startBtn}
        >
          Начать работу с обращениями
        </Button>
      </div>
    </div>
  );
};
