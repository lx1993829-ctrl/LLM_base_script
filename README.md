Install jtop outside conda environment:
sudo pip install -U jetson-stats

Then activate conda environment and install jtop:
pip install -U jetson-stats

Test jtop by type "jtop" at terminal, it may return "The jtop.service is not active. Please run: sudo systemctl restart jtop.service". Then input "sudo systemctl restart jtop.service" as suggested.
