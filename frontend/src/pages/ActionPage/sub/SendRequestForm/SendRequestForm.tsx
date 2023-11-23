import { Textarea, TextareaProps } from '@mantine/core';

import { useAppStore } from '../../../../store';
import { ActionPageFormProps } from '../../../../shared';

import styles from './SendRequestForm.module.scss';

const TEXTAREA_CONFIG: TextareaProps = {
  minRows: 6,
  maxRows: 12,
  autosize: true,
};

export const SendRequestForm = ({ className }: ActionPageFormProps) => {
  const { setRequest, request } = useAppStore();

  return (
    <div className={className}>
      <h2 className="title">Заполните обращение</h2>
      <Textarea
        onChange={(e) => setRequest(e.target.value)}
        value={request}
        variant="filled"
        label="Введите обращение гражданина"
        placeholder="Обращение"
        className={styles.textarea}
        {...TEXTAREA_CONFIG}
      />
    </div>
  );
};
