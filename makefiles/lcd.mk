# lcd.%: Generic lcd make command
#lcd.%:
#	${PYTHON} scripts/lcd/$*.py

## lcd.calibrate.% : Generic lcd calibrate command
lcd.calibrate.%:
	${PYTHON} scripts/lcd/calibrate/$*.py $(HYDRA_FLAGS)

## lcd_calibrate: Run all calibration steps
LCD_CALIBRATE_DEPS := lcd.calibrate.decode_correspondences
LCD_CALIBRATE_DEPS += lcd.calibrate.get_intrinsic_extrinsic
lcd_calibrate: $(LCD_CALIBRATE_DEPS)

