import axios from "axios";

const api_base = "http://127.0.0.1:5000";

export async function postRequest(api_suffix, post_data) {
    try {
        // 使用 axios 发送 POST 请求
        const response = await axios.post(api_base + api_suffix, post_data, {
            headers: {
                "Content-Type": "application/json",
            },
        });
        // response.data 是服务器返回的数据
        console.log('成功: ', api_suffix, response);
        return response;
    } catch (error) {
        console.log('失败: ', api_suffix, error);
        return error.response;
    }
}

export async function getRequest(api_suffix){
    try {
        // 使用 axios 发送 POST 请求
        const response = await axios.get(api_base + api_suffix);
        // response.data 是服务器返回的数据
        console.log('成功: ', api_suffix, response);
        return response;
    } catch (error) {
        console.log('失败: ', api_suffix, error);
        return error.response;
    }
}

import OpenAI from 'openai';

const deepseek = new OpenAI({
    baseURL: 'https://api.deepseek.com',
    apiKey: 'sk-8ba2fc7ed5134e398b0bb030ac9c7412',
    dangerouslyAllowBrowser: true,
});

export async function deepseekRequest(messages, model) {
    // messages 是一个数组，每个元素是一个对象
    // 每个对象有一个 key 是 role，值是 system 或者 user 或者 assistant
    // 另一个 key 是 content，值是字符串
    // 例如 [{ role: "system", content: "You are a helpful assistant." }, {role: 'user', content: '你好'}, {role: 'assistant', content: '你好'}]
    try {
        // 使用 axios 发送 POST 请求
        const completion = await deepseek.chat.completions.create({
            messages: messages,
            model: model,
          });
        console.log('deepseekRequest 成功: ', completion);
        return completion;
        
    } catch (error) {
        console.log('deepseekRequest 失败: ', error);
        return error;
    }
}
