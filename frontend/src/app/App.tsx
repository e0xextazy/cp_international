import { useState } from 'react';
import { MantineProvider, createTheme } from '@mantine/core';

import '@mantine/core/styles.css';
import '../index.scss';
import styles from './App.module.scss';

import { StartScreenPage } from '../pages/StartScreenPage/StartScreenPage';
import { Header } from '../widgets/Header/Header';
import { Footer } from '../widgets/Footer/Footer';
import { ActionPage } from '../pages/ActionPage/ActionPage';
import { AppPages } from './config';

const theme = createTheme({});

export function App() {
  const [page, setPage] = useState(AppPages.START_PAGE);

  const getPage = () => {
    if (page === AppPages.ACTION_PAGE) {
      return <ActionPage onNav={setPage} />;
    }

    return <StartScreenPage onNav={setPage} />;
  };

  return (
    <MantineProvider defaultColorScheme="dark" theme={theme}>
      <Header />
      <div className={styles.layout}>{getPage()}</div>
      <Footer />
    </MantineProvider>
  );
}
