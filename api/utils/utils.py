import base64


def norm_flow(flow):
    try:
        request = base64.b64decode(flow).decode("utf-8")
        request = request.replace('\r\n', '\n')
        # 提取请求正文（如果存在）
        body_index = request.find('\n\n')
        if body_index == -1:
            request += "\n\n"
        body_index = request.find('\n\n')

        # print(body_index)
        body = ''
        if body_index != -1:
            body = request[body_index + 2:].lstrip()

        prior = request[:body_index + 1]
        # 使用换行符分割请求

        request_lines = prior.strip().split('\n')

        # 提取请求方法、路径和HTTP版本
        tmp = []
        tmp_i = 0
        for i in range(0, len(request_lines)):
            tmp.append(request_lines[i])
            if ' http/' in request_lines[i].lower():
                tmp_i = i
                break

        first_line = '%0a'.join(tmp)

        # 提取请求头部
        headers = ["Host: xxxxx:80"]
        for line in request_lines[tmp_i:]:
            if line.strip() == '':
                break
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            tmp_key = key.strip().lower()
            tmp_value = value.strip().lower()
            if "stgw" in tmp_key or "waf" in tmp_key or "scanner" in tmp_key or tmp_key == "host":
                continue
            if tmp_value == "tag":
                continue
            headers.append("{}: {}".format(key.strip(), value.strip()))

        new_http = "{}\n{}\n\n{}".format(first_line, "\n".join(headers), body)
        return new_http.replace('\n', '\r\n')
    except:
        return ""

