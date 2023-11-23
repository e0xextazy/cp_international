
import { useGlitch } from 'react-powerglitch'
import { IconArrowRight } from '@tabler/icons-react';
import { Button } from "@mantine/core";

import styles from "./StartScreenPage.module.scss";

export const StartScreenPage = () => {
    const glitch = useGlitch();

    return (
        <div className={styles.page}>
            <div className={styles.banner}>
                <div className={styles.logo}>
                    <h1>
                        AppealClassifier
                    </h1>
                </div>
                <span ref={glitch.ref}>Powerful AI powered tool</span>
            </div>
            <div className={styles.action}>
                <Button
                    fullWidth
                    variant="light"
                    justify="space-between"
                    rightSection={<IconArrowRight />}
                    className={styles.startBtn}>
                    Начать работу с обращениями
                </Button>
            </div>
        </div>
    )
}
