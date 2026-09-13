# OpenDPD 2.2.2 — Report bugs from Studio

Try [OpenDPD Studio](https://opendpd.com/studio/) or install `pip install -U "opendpd[gui]==2.2.2"`.

- Add **Report bugs** to the GUI toolbar, including a compact mobile control. It is also available before a session starts and when the compute server cannot be reached.
- Open a description box with keyboard focus ready for typing. **Continue on GitHub** opens the OpenDPD repository's new-issue form in a separate tab, with the description and public Studio version filled in. The user reviews and submits the issue on GitHub; the running experiment stays open.
- Keep the draft when the dialog is dismissed, restore keyboard focus to the toolbar, and handle descriptions that are too long to pass through GitHub's URL form.
- Translate the reporting flow into all nine interface languages. Workspace paths, tokens, datasets, logs and experiment details are not automatically attached.

GitHub's own form controls its initial keyboard focus and currently focuses the title. Studio's small description dialog provides reliable focus without depending on GitHub's generated element IDs or cross-origin scripting. GitHub sign-in may be required to submit an issue.

CUDA training, 10/150-epoch defaults, preview cadence, upload validation, isolation and 24-hour cleanup are unchanged from 2.2.1.
