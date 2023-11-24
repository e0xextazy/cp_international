import { Slider } from '@mantine/core';

import styles from './ThresholdSlider.module.scss';

import { useAppStore } from '../../store';

export const ThresholdSlider = () => {
  const { confidenceThreshold, setConfidenceThreshold } = useAppStore();

  return (
    <div className={styles.slider}>
      <Slider
        color="blue"
        onChange={setConfidenceThreshold}
        value={confidenceThreshold}
        marks={[
          { value: 0, label: '0%' },
          { value: 50, label: '50%' },
          { value: 75, label: '75%' },
          { value: 95, label: '95%' },
        ]}
      />
    </div>
  );
};
