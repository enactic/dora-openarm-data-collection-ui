// Not on every event but only on a change. Rebuilding the list clears
// a selection, which the operator may make to copy an error.
let lastErrors = null;

eventSource.addEventListener("errors", (e) => {
  if (e.data === lastErrors) return;
  lastErrors = e.data;
  const errors = JSON.parse(e.data);
  document.getElementById("errors").hidden = errors.length === 0;
  document.getElementById("error-mark").hidden = errors.length === 0;
  document.getElementById("error-list").replaceChildren(
    ...errors.map((error) => {
      const meta = document.createElement("span");
      meta.className = "meta mono";
      meta.textContent = `[${error.time}] ${error.source}`;
      if (error.count > 1) meta.textContent += ` ×${error.count}`;
      const item = document.createElement("li");
      // A space for copying.
      item.append(meta, " ", error.message);
      return item;
    }),
  );
});
