"use client";

import { animate } from "framer-motion";
import { useEffect, useState } from "react";

interface AnimatedNumberProps {
  value: number;
  format?: (value: number) => string;
}

export function AnimatedNumber({ value, format = (value) => value.toLocaleString() }: AnimatedNumberProps) {
  const [displayValue, setDisplayValue] = useState(() => format(0));

  useEffect(() => {
    const controls = animate(0, value, {
      duration: 0.9,
      onUpdate(latest) {
        setDisplayValue(format(latest));
      },
    });

    return () => controls.stop();
  }, [value, format]);

  return <span>{displayValue}</span>;
}
