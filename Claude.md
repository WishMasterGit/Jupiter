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

## Claude workstream
