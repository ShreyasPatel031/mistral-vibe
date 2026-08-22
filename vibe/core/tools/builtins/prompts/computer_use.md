Drive a real Chromium browser to complete a task on a public website.

Use this when a task requires *acting* on a site rather than just reading it: applying
filters, filling forms, picking dates, adding items to a cart, stepping through a signup
flow, or checking whether a user journey actually works. The agent sees rendered
screenshots plus the DOM and clicks, types, and scrolls like a person would.

Prefer `web_fetch` when you only need the text of a page — it is far cheaper and faster.
Reach for `computer_use` only when interaction is the point.

Parameters:

- `url` — the public URL to open first. http/https only.
- `task` — what to accomplish, in concrete checkable terms. "Filter to 2-bedroom rentals
  under $2500 and report the top 3" works far better than "look at rentals".
- `intent` — optional. Who the user is and what they are trying to achieve, so the agent
  behaves like that person. Useful for usability checks.
- `max_steps` — optional override of the configured step budget.

Notes:

- Each step is one browser action, and long journeys need budget. If the result comes back
  with `budget_exhausted` set, the agent was cut off rather than finished — re-run with a
  higher `max_steps` before concluding the site cannot do the thing.
- The result reports `completed`, the `final_url`, a step-by-step trace of actions and the
  agent's reasoning, and a `summary` of what it achieved or could not achieve.
- Sites with aggressive bot protection may show a CAPTCHA or block the browser outright.
  That surfaces in the trace; it is an environment limit, not a task failure.
- Requires the optional `browser-use` dependency and a Chromium install. The tool hides
  itself when either is missing.
