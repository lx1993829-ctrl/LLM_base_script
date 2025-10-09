Install jtop outside conda environment:
sudo pip install -U jetson-stats

Then activate conda environment and install jtop:
pip install -U jetson-stats

Test jtop by type "jtop" at terminal, it may return "The jtop.service is not active. Please run: sudo systemctl restart jtop.service". Then input "sudo systemctl restart jtop.service" as suggested.

Example:
python ./run_tests.py --run_input ./input.txt --iterations 1 --model_path /home/jetson/llama-2-7b-chat --w_bit 4 --q_group_size 128 --load_quant /home/jetson/llama-2-7b-chat/quant_cache/llama2-7b-chat-w4-g128-awq-v2.pt
