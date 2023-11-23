import { Avatar, Indicator } from '@mantine/core';
import styles from './Header.module.scss';
import { HackathonLogo } from '../HackathonLogo/HackathonLogo';

export const Header = () => {
  return (
    <div className={styles.header}>
      <HackathonLogo />
      <div className={styles.profile}>
        <span>5random team</span>
        <Indicator size="6" processing color="yellow">
          <Avatar radius="xs" src="images/team-logo.jpg" />
        </Indicator>
      </div>
    </div>
  );
};
