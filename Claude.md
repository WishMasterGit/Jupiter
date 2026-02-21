# Planet imagery processing tool

## What is it about

this is planet image/video processing tool to achieve crisp results from the earth based telescopes

## Technical decisions

- Main language rust
- Try to use constants from std for inline numbers
  - if constants aren't available in the std try checking libraries for corresponding constants
  - if there are no libraries, create constants so it is easier to understand the meaning (when it applicable)
- based on research lets use egui as our UI rendering engine to keep it crossplatform
- app should be crossplatform Windows/Mac/Linux but not mobile phones
- write tests for all jupiter-core logic

## Code Style

- For the cases when we need to use "match" to branch the logic, lets avoid numeric indexes in matching and instead use named enums

- Prefer splitting logic into separate files under the folder, for example when you need to implement multiple different worker handlers, create workers folder and make ##\_hanlder.rs for different handlers instead of putting everything into one file

- mod files should only be used to define module structure and re-exporting

## Claude workstream

- make sure to build with features gpu and normally while testing
- place tests in separate files in "tests" folder
- save all researches in the docs folder
