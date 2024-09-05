# IVT Cross Sections

---

This repository runs calculations and plots for the CW3E IVT Cross Sections for GFS and ECMWF. 

Current capabilities includes meridional transects from 25°N to 60°N, every 5° longitude from 175°W to 105°W
with a forecast lead every 12 hours out to a 120 hour lead.

## To run:

---

To run all longitudinal cross sections for each model:

```bash
## runs plots for GFS
singularity exec --bind /data:/data,/home:/home,/work:/work,/common:/common -e /data/projects/containers/ivt_cross_sections/ivt_cross_sections.sif /opt/conda/bin/python /data/projects/operations/ivt_cross_sections/run_tool.py "GFS"

## runs plots for ECWMF
singularity exec --bind /data:/data,/home:/home,/work:/work,/common:/common -e /data/projects/containers/ivt_cross_sections/ivt_cross_sections.sif /opt/conda/bin/python /data/projects/operations/ivt_cross_sections/run_tool.py "ECMWF"
```