# Makefile for NPU TTS Research
#
# Targets:
#   make figures    — regenerate all paper figures (requires data in data/m4_real/)
#   make paper      — compile LaTeX paper to PDF
#   make clean      — remove generated files

PYTHON ?= python
LATEX ?= pdflatex

FIGURE_DIR = paper/figures
FIGURE_SCRIPTS = $(wildcard $(FIGURE_DIR)/gen_*.py)
FIGURE_PNGS = $(patsubst $(FIGURE_DIR)/gen_%.py,$(FIGURE_DIR)/fig_%.png,$(FIGURE_SCRIPTS))

.PHONY: figures paper clean install

# Install Python dependencies
install:
	pip install -r requirements.txt

# Generate all figures
# Note: gen_precision_curve.py and gen_spectrograms.py require raw data in data/m4_real/
# Other figures (architecture, timing, waterfall, trajectory) use embedded data and always work
figures:
	$(PYTHON) $(FIGURE_DIR)/gen_architecture_diagram.py
	$(PYTHON) $(FIGURE_DIR)/gen_timing_breakdown.py
	$(PYTHON) $(FIGURE_DIR)/gen_error_waterfall.py
	$(PYTHON) $(FIGURE_DIR)/gen_optimization_trajectory.py
	@echo "--- Figures with embedded data generated ---"
	@echo "For gen_precision_curve.py and gen_spectrograms.py, set NPU_DATA_DIR to your data directory"

# Compile LaTeX paper
paper:
	cd paper && $(LATEX) paper.tex && $(LATEX) paper.tex
	@echo "--- paper/paper.pdf generated ---"

# Run precision comparison (requires data)
compare:
	$(PYTHON) methodology/compare_precision.py

# Run audio comparison (requires data + ONNX model)
audio:
	$(PYTHON) methodology/audio_compare.py

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -f _tmp_*.onnx
