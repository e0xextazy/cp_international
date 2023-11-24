import { Badge, Button, Fieldset, Textarea } from '@mantine/core';

import styles from './Response.module.scss';

import { ActionPageFormProps } from '../../../../shared';
import { TagsKeys, useAppStore } from '../../../../store';
import { IconAlertCircleFilled } from '@tabler/icons-react';

const TRANSLATE_CONFIG: Record<TagsKeys, string> = {
  LOC: 'Локация',
  ADDRESS: 'Адрес',
  DATE: 'Дата',
  MONEY: 'Деньги',
  ORG: 'Организация',
  PHONE: 'Телефон',
  PER: 'ФИО',
};

export const Response = ({ className }: ActionPageFormProps) => {
  const { response, setResponse } = useAppStore();

  const sendToModeration = () => {
    setResponse(null);
  };

  const { executor = '', subtopic = '', topic = '' } = response!;

  const hasUndefinedValues = Object.values(response?.tags || [true]).some((v) => Boolean(v));
  const tags = Object.entries(response?.tags || [])
    .filter(([, val]) => Boolean(val?.length))
    .map(([key, val]) => `${TRANSLATE_CONFIG[key?.toUpperCase() as TagsKeys]}: ${val.join(', ')}`);

  const showModerationButton = hasUndefinedValues;

  const actionBtnText = showModerationButton ? 'Отправить на ручную модерацию' : 'Подтвердить и отправить';

  return (
    <div className={className}>
      <h2 className="title">Результаты</h2>
      <Fieldset className={styles.fieldList}>
        <Textarea label="Исполнитель" value={executor} error={!Boolean(executor)} />
        <Textarea label="Группа тем" value={topic} error={!Boolean(topic)} />
        <Textarea label="Тема" value={subtopic} error={!Boolean(subtopic)} />
        <div className={styles.metricsWrapper}>
          <h2>Именованные сущности</h2>
          <div className={styles.metrics}>
            {tags.map((tag, idx) => (
              <Badge variant="outline" className={styles.metric} key={idx}>
                {tag}
              </Badge>
            ))}
          </div>
        </div>
        <div className={styles.alert}>
          {showModerationButton && (
            <span>
              <IconAlertCircleFilled />
              Модель не уверена в результатах выдачи
            </span>
          )}
          <Button
            color={showModerationButton ? undefined : 'green'}
            size="md"
            className={styles.moderateBtn}
            onClick={sendToModeration}
          >
            {actionBtnText}
          </Button>
        </div>
      </Fieldset>
    </div>
  );
};
