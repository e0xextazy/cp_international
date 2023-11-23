import { ActionPageFormProps } from '../../../../shared';
import styles from './Response.module.scss';

export const Response = ({ className }: ActionPageFormProps) => {
  return (
    <div className={className}>
      <h2 className="title">Результаты</h2>
    </div>
  );
};
