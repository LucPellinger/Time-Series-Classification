# Migration Checklist

## Pre-Migration
- [x] Backup all current files
- [x] Document current working directory structure
- [x] Note any hardcoded paths in the code

## Migration Steps
- [x] Run create_project_structure.sh to create directories
- [x] Run migration_map.py to move files
- [x] Run update_imports.py to fix import statements
- [ ] Create setup.py and requirements.txt
- [ ] Create __init__.py files with proper exports

## Post-Migration
- [ ] Update any configuration files with new paths using paths.py
- [ ] Update notebook imports if you have any
- [ ] Test basic imports: `python -c "from time_series_classification import setup_logger"`
- [ ] Run a simple training script to verify everything works
- [ ] Update documentation/README with new structure

## Clean Up
- [ ] Remove old files from original location (after confirming everything works)
- [ ] Update .gitignore for new structure
- [ ] Commit the reorganized structure to version control