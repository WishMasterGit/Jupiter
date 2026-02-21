# TODO

- mod files should only be used to define module structure and re-exporting, all the logic should be moved to separate file with proper name [Done]
- Reading, Writing, Cropping and Debayering are not part of the pipeline and should be independent [Skip]
- in jupiter-gui, steps of pipeline should have numbers, for example 1. Frame selection, 2. Alignment etc [Done]
- only first step of pipeline should be active when user opens the file, all others should be visible but disabled, including settings [Done]
- lets refactor statuses used to control pipline [Done]
- when user triggres previous steps of pipline, further steps should be disabled and enabled as user proceeds through the pipeline [Done]
- make font of the bottom panel larger [Done]
- when pipeline stage finished header should get green background not a button [Done]
- "alignment keep" percentage setting should be in percents in gui not in fractions [Done]
- sharpening should be applied on mouse up not on time delay [Done]
- background for unprocessed stage should be transparent not orange [Done]
- for frame selection small graph should be shown that represents frames quality
- for multipoint alignment there should be option to automatically choose appropriate AP size
- there should be option to switch between processed and raw frames, also when user changes raw frames it should only update the original frames preview
- for log preview it should prerender 4 lines to avoid screen jumping when new log appears, user should also be able to scroll through the log
