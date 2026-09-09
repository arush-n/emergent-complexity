function toggleButton(container, count) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "count-toggle";
  button.dataset.count = String(count);
  button.setAttribute("aria-pressed", "false");
  button.textContent = String(count);
  container.appendChild(button);
  return button;
}

function enabledCounts(container) {
  return Array.from(container.querySelectorAll(".count-toggle[aria-pressed='true']"))
    .map((button) => Number(button.dataset.count));
}

function setCounts(container, counts) {
  const enabled = new Set(counts);
  container.querySelectorAll(".count-toggle").forEach((button) => {
    const active = enabled.has(Number(button.dataset.count));
    button.setAttribute("aria-pressed", String(active));
  });
}

export function createRuleEditor(
  birthContainer,
  survivalContainer,
  onChange,
  { maxCount = 8, separator = "" } = {},
) {
  for (let count = 0; count <= maxCount; count += 1) {
    toggleButton(birthContainer, count);
    toggleButton(survivalContainer, count);
  }

  const emit = () => {
    const birth = enabledCounts(birthContainer);
    const survival = enabledCounts(survivalContainer);
    onChange(`B${birth.join(separator)}/S${survival.join(separator)}`);
  };

  [birthContainer, survivalContainer].forEach((container) => {
    container.addEventListener("click", (event) => {
      const button = event.target.closest(".count-toggle");
      if (!button) return;
      const active = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", String(!active));
      emit();
    });
  });

  return {
    getRule() {
      const birth = enabledCounts(birthContainer);
      const survival = enabledCounts(survivalContainer);
      return `B${birth.join(separator)}/S${survival.join(separator)}`;
    },
    setRule(rule, { emitChange = false } = {}) {
      const parts = rule.trim().toUpperCase().split("/");
      if (parts.length !== 2 || !parts[0].startsWith("B") || !parts[1].startsWith("S")) {
        throw new Error(`Use the form B<counts>/S<counts>, with counts from 0 to ${maxCount}.`);
      }
      const parseCounts = (text) => {
        if (!text) return [];
        let counts;
        if (!separator) counts = [...text].map(Number);
        else if (text.includes(",")) counts = text.split(",").map(Number);
        else if (text.length === 2 && Number(text) >= 10 && Number(text) <= maxCount) counts = [Number(text)];
        else if (text.length === 2) throw new Error(`Two-digit counts must be between 10 and ${maxCount} or comma-separated.`);
        else counts = [...text].map(Number);
        if (
          counts.some((count) => !Number.isInteger(count) || count < 0 || count > maxCount) ||
          new Set(counts).size !== counts.length
        ) {
          throw new Error(`Counts must be unique integers from 0 to ${maxCount}.`);
        }
        return counts;
      };
      setCounts(birthContainer, parseCounts(parts[0].slice(1)));
      setCounts(survivalContainer, parseCounts(parts[1].slice(1)));
      if (emitChange) emit();
    },
  };
}
