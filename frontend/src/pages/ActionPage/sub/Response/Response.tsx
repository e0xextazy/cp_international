import { Badge, Button, Fieldset, Textarea } from '@mantine/core';

import { ActionPageFormProps } from '../../../../shared';
import styles from './Response.module.scss';

export const Response = ({ className }: ActionPageFormProps) => {
  const showModerationButton = false;

  return (
    <div className={className}>
      <h2 className="title">Результаты</h2>
      <Fieldset className={styles.fieldList}>
        <Textarea label="Куда направить обращение" />
        <Textarea label="Куда направить обращение" />
        <Textarea label="Куда направить обращение" />
        <div className={styles.metricsWrapper}>
          <h2>Метрики</h2>
          <div className={styles.metrics}>
            <Badge variant="outline" className={styles.metric}>
              МВД
            </Badge>
            <Badge variant="outline" className={styles.metric}>
              ЧП
            </Badge>
            <Badge variant="outline" className={styles.metric}>
              МВД
            </Badge>
          </div>
        </div>
        {showModerationButton && (
          <Button size="md" className={styles.moderateBtn}>
            Отправить на модерацию
          </Button>
        )}
      </Fieldset>
    </div>
  );
};
