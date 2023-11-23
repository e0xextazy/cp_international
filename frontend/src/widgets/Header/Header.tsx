import { Avatar, Indicator } from '@mantine/core';
import styles from './Header.module.scss';
import { HackathonLogo } from '../HackathonLogo/HackathonLogo';

export const Header = () => {
  return (
    <div className={styles.header}>
      <HackathonLogo />
      <div className={styles.profile}>
        <span>5random</span>
        <Indicator size="6" processing>
          <Avatar
            radius="xs"
            src="https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-1.png"
          />
        </Indicator>
      </div>
    </div>
  );
};
