import styles from './HackathonLogo.module.scss';

export const HackathonLogo = () => {
  return (
    <div className={styles.logo}>
      <img src="images/cp-logo.png" alt="Логотип хакатона" />
    </div>
  );
};
