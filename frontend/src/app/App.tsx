import { MantineProvider, createTheme } from '@mantine/core';

import '@mantine/core/styles.css';
import '../index.scss';

import styles from './App.module.scss';
import { StartScreenPage } from '../pages/StartScreenPage/StartScreenPage';
import { Header } from '../widgets/Header/Header';
import { Footer } from '../widgets/Footer/Footer';

const theme = createTheme({});

export function App() {
  return (
    <MantineProvider defaultColorScheme="dark" theme={theme}>
      <Header />
      <div className={styles.layout}>
        <StartScreenPage />
      </div>
      <Footer />
    </MantineProvider>
  );
}
