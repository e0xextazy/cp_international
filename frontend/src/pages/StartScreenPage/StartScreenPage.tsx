import { useGlitch } from 'react-powerglitch';
import { IconArrowRight } from '@tabler/icons-react';
import { Button } from '@mantine/core';

import styles from './StartScreenPage.module.scss';
import { PageProps } from '../../shared/page';
import { AppPages } from '../../app/config';

export const StartScreenPage = ({ onNav }: PageProps) => {
  const glitch = useGlitch();

  const goToAction = () => {
    onNav(AppPages.ACTION_PAGE);
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
