from bs4 import BeautifulSoup, Tag
from sklearn.cluster import DBSCAN
import numpy as np
import math
from typing import List, Dict, Union, Callable, List, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
import time

from concurrent.futures import ThreadPoolExecutor, as_completed


#reference:https://github.com/huggingface/chat-ui/blob/main/src/lib/server/websearch/scrape/parser.ts

def distance_function(a: Tag, b: Tag) -> float:
    rect1 = a['rect']
    rect2 = b['rect']
    dx = 0
    dy = 0
    if rect1['x'] + rect1['width'] < rect2['x']:
        dx = rect2['x'] - (rect1['x'] + rect1['width'])
    elif rect2['x'] + rect2['width'] < rect1['x']:
        dx = rect1['x']  - (rect2['x']  + rect2['width'])
    if rect1['y'] + rect1['height'] < rect2['y']:
        dy = rect2['y'] - (rect1['y'] + rect1['height'] )
    elif rect2['y'] + rect2['height']  < rect1['y']:
        dy = rect1['y'] - (rect2['y'] + rect2['height'] )
    distance = math.sqrt(dx * dx + dy * dy)
    return distance

def is_only_child(node: Tag) -> bool:
    if not node.parent:
        return True
    if node.parent.name == "body":
        return False
    if len(node.parent.contents) == 1:
        return True
    return False


def get_all_readable_nodes(driver: webdriver.Chrome, timeout_seconds = 20) -> Dict[WebElement, Dict[str, str]]:
    def get_parent(element:WebElement, parent_cache:Dict):
        """获取给定元素的父节点"""
        if element in parent_cache:
            return parent_cache[element],parent_cache
        parent = driver.execute_script("return arguments[0].parentNode;", element)
        parent_cache[element] = parent
        return parent,parent_cache
    def is_visible(element)-> bool:
        if element is None:
            return False
        return element.is_displayed() and element.value_of_css_property('opacity') != '0'


    def is_only_child(element:WebElement, parent_cache:Dict):
        parent,_ = get_parent(element, parent_cache)
        if not parent:
            return True
        if parent.tag_name == "BODY":
            return False
        return len(driver.execute_script("return arguments[0].children;", parent)) == 1

    def has_valid_inline_parent(element:WebElement, parent_cache:Dict):
        parent,_ = get_parent(element, parent_cache)
        return parent and parent.tag_name.lower() not in ['div', 'section', 'article', 'main', 'body']
    
    def find_the_top(element:WebElement, parent_cache:Dict, max_depth=5):
        depth = 0
        top_element = None

        while depth < max_depth:
            parent, parent_cache = get_parent(element, parent_cache)

            # 如果找到了 'main' 或 'body'，或者没有父节点，则直接返回
            if not parent or parent.tag_name.lower() in ['main', 'body']:
                top_element = parent
                break
            
            element = parent
            depth += 1

        # 如果没有找到 'main' 或 'body' 且达到了最大深度，返回None
        return top_element, parent_cache
    def find_code_container(element:WebElement, parent_cache:Dict):
        """
        寻找所有可能作为 <code> 元素父节点的 <pre> 和 <p> 元素。
        """
        # 通过JavaScript获取当前元素的父节点
        parent, parent_cache = get_parent(element, parent_cache)
        
        # 检查父节点是否是 <ul> 或 <ol>
        if parent.tag_name.lower() in ['pre', 'p']:
            return parent, parent_cache
        else:
            return None, parent_cache
    def find_list_container(element:WebElement, parent_cache:Dict):
        """
        寻找列表容器作为 <li> 元素的父节点。
        """
        # 通过JavaScript获取当前元素的父节点
        parent, parent_cache = get_parent(element, parent_cache)
        
        # 检查父节点是否是 <ul> 或 <ol>
        if parent.tag_name.lower() in ['ul', 'ol']:
            return parent, parent_cache
        else:
            return None, parent_cache

    def find_table_row(element:WebElement, parent_cache:Dict):
        """
        寻找表格行作为 <td> 或 <th> 元素的父节点。
        """
        # 通过JavaScript获取当前元素的父节点
        parent, parent_cache = get_parent(element, parent_cache)
        
        # 检查父节点是否是 <tr>
        if parent.tag_name.lower() == 'tr':
            return parent, parent_cache
        else:
            return None, parent_cache

    def find_table_container(element:WebElement, parent_cache:Dict):
        """
        寻找表格容器作为 <tr> 元素的父节点。
        """
        # 通过JavaScript获取当前元素的父节点
        parent, parent_cache = get_parent(element, parent_cache)
        
        # 检查父节点是否是 <table>, <thead>, <tbody>, 或 <tfoot>
        if parent.tag_name.lower() in ['table', 'thead', 'tbody', 'tfoot']:
            return parent, parent_cache
        else:
            return None, parent_cache
    def find_highest_direct_parent_of_readable_node(element,parent_cache:Dict):
        """
        寻找给定元素的最高级可读节点的直接父节点。
        """
    
        # 寻找第一个不满足是唯一子节点或者父节点无效的节点
        timer = time.time()
        max_depth = 3
        current_depth = 0
        timer = time.time()
        while has_valid_inline_parent(element, parent_cache) and is_only_child(element, parent_cache) and current_depth < max_depth:
            # 通过JavaScript获取当前元素的父节点
            parent, parent_cache = get_parent(element, parent_cache)
            element = parent
    
            # 增加计数器
            current_depth += 1

        time_0 = time.time()
        # 继续寻找直到找到一个不是唯一子节点的节点或者达到DOM树的根节点
        while element and is_only_child(element, parent_cache):
            # 如果当前节点的父节点无效，则退出循环
            if not has_valid_inline_parent(element, parent_cache):
                break
            # 通过JavaScript获取当前元素的父节点
            element, parent_cache = get_parent(element, parent_cache)
        # 如果找到断开连接的节点，则抛出异常
        if not element:
            raise Exception("Disconnected node found, this should not be possible when traversing through the DOM")
    
        # 根据找到的节点类型进行不同的处理
        time_0 = time.time()
        tag_name = element.tag_name.lower()
        if tag_name in ['span', 'code', 'div']:
            # 对于span、code和div标签，需要继续向上查找父节点
            code_container,_ = find_code_container(element, parent_cache)
            if code_container:
                element = code_container
            #如果不是，那么向上一直搜索到最顶端
            else:
                element,_ = find_the_top(element, parent_cache)
        elif tag_name == 'li':
            # 对于li标签，找到其列表容器作为父节点
            list_container,_ = find_list_container(element, parent_cache)
            if list_container:
                element = list_container
            else:
                element,_ = find_the_top(element, parent_cache)
        elif tag_name in ['td', 'th']:
            # 对于td和th标签，找到其表格行作为父节点
            table_row,_ = find_table_row(element, parent_cache)
            if table_row:
                element = table_row
            else:
                element,_ = find_the_top(element, parent_cache)
        elif tag_name == 'tr':
            # 对于tr标签，找到其表格容器作为父节点
            table_container,_ = find_table_container(element, parent_cache)
            if table_container:
                element = table_container
            else:
                element,_ = find_the_top(element, parent_cache)
        # 返回找到的最高级可读节点的直接父节点
        time_1 = time.time()
        #print(f"time_0: {time_0 - timer}s, time_1: {time_1 - time_0}s, time_total: {time_1-timer}s")
        return element,parent_cache
    
    def is_prohibited_node(node):
        # 获取节点的 class 属性
        class_attr = node.get_attribute('class')

        # 如果 class 属性不存在，则返回 False
        if not class_attr:
            return False

        # 将 class 属性按空格分割成列表
        classes = class_attr.split()

        # 定义禁止的类名列表
        prohibited_classes = ["footer", "nav", "aside", "script", "style", "noscript", "form", "button"]

        # 检查是否有禁止的类名
        return any(cls in classes for cls in prohibited_classes)
    def does_node_pass_heuristics(element:WebElement,parent_cache:Dict):
        try:
            timer = time.time()
            # 检查文本长度
            #if not isinstance(element.text, str):
            #    raise TypeError("Expected 'text' attribute to be a string")
            # 查找最高级别的直接可读父节点
            
            parent,parent_cache = find_highest_direct_parent_of_readable_node(element,parent_cache)
            time_0 = time.time()
            # 检查父节点是否可见
            if not is_visible(parent):
                return False, None,parent_cache
            # 获取父节点的矩形信息
            rect = getattr(parent, 'rect', None)
            if rect is None:
                raise AttributeError("Parent node does not have a 'rect' attribute")

            # 获取页面的滚动位置
            scroll_x = driver.execute_script("return window.scrollX;")
            scroll_y = driver.execute_script("return window.scrollY;")

            # 计算元素相对于整个页面的位置
            x_page = rect['x'] + scroll_x
            y_page = rect['y'] + scroll_y

            # 检查矩形尺寸
            if rect['width'] < 4 or rect['height'] < 4:
                return False, None,parent_cache

            # 禁止节点检查
            if is_prohibited_node(parent):
                return False, None
            time_1 = time.time()

            return True, {'node': parent, 'rect': {'x': x_page, 'y': y_page, 'width': rect['width'], 'height': rect['height']}},parent_cache

        except Exception as e:
            print(f"An error occurred: {e}")
            return False, None,parent_cache


    #函数主体部分
    parent_cache = {}
    readable_nodes = []

    potential_text_containers = driver.find_elements(By.XPATH, """
                                                            //body//p[.//text()[string-length(.) > 10]] |
                                                            //body//span[.//text()[string-length(.) > 10]] |
                                                            //body//div[.//text()[string-length(.) > 10]] |
                                                            //body//li[.//text()[string-length(.) > 10]] |
                                                            //body//td[.//text()[string-length(.) > 10]]
                                                            """)

    print(f"find {len(potential_text_containers)} text containers with more than 10 characters")

    def process_element(element, parent_cache):
        result, node_data, updated_parent_cache = does_node_pass_heuristics(element, parent_cache)
        # 更新缓存
        parent_cache.update(updated_parent_cache)
        return (result, node_data)

    # 定义超时时间（秒）
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交任务并获取Future对象
        futures = {executor.submit(process_element, container, parent_cache.copy()): container for container in potential_text_containers}

        completed_results = []

        for future in as_completed(futures, timeout=None):  # 使用None以避免超时错误
            if time.time() - start_time >= timeout_seconds:
                # 取消未完成的任务
                for f in futures.keys():
                    if not f.done():
                        f.cancel()
                break
            try:
                # 获取已完成的任务结果
                result = future.result()
                completed_results.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    readable_nodes = [node_data for result, node_data in completed_results if result]

    print(f"find {len(readable_nodes)} readable nodes")

    # 如果超时，打印一条消息并返回当前收集到的结果
    if time.time() - start_time >= timeout_seconds:
        print("Timeout occurred; returning current results.")

    print(f"find {len(readable_nodes)} readable nodes")
    unique_parents = []
    seen_elements = set()
    for node in readable_nodes:
        if node['node'] and node['node'] not in seen_elements:
            seen_elements.add(node['node'])
            unique_parents.append(node)
    print(f"find {len(unique_parents)} unique parents")
    return unique_parents


def cluster_readable_nodes(readableNodes):
    # 计算所有节点之间的距离矩阵
    num_nodes = len(readableNodes)
    distances = np.zeros((num_nodes, num_nodes))
    #distances = np.zeros((num_nodes,2))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distances[i, j] = distance_function(readableNodes[i], readableNodes[j])
            distances[j, i] = distances[i, j]  # 因为距离是对称的
        
    # 使用 DBSCAN 进行聚类
    # eps 控制节点之间的最大距离，min_samples 控制一个聚类中至少需要的样本数
    # 注意：这里使用 'precomputed' 指定距离矩阵
    dbscan = DBSCAN(eps=0.5, min_samples=1, metric='precomputed').fit(distances)
    #dbscan = DBSCAN(eps=10, min_samples=1).fit(distances)
    
    # 根据聚类标签创建聚类
    clusters = {}
    for i, label in enumerate(dbscan.labels_):
        if label == -1:  # 噪声点
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return list(clusters.values()), [i for i, label in enumerate(dbscan.labels_) if label == -1]


import re

def total_text_length(nodes, cluster: List[int]) -> int:
    whitespace_pattern = r'\s+'
    return sum([len(re.sub(whitespace_pattern, '', nodes[i]["node"].text)) for i in cluster])

def approximately_equal(a: float, b: float, epsilon: float = 1) -> bool:
    return abs(a - b) < epsilon


def get_cluster_bounds(nodes, cluster: List[int]) -> Dict:
    x_values = (nodes[i]["rect"]['x'] for i in cluster)
    y_values = (nodes[i]["rect"]['y'] for i in cluster)
    width_values = (nodes[i]["rect"]['x'] + nodes[i]["rect"]['width'] for i in cluster)
    height_values = (nodes[i]["rect"]['y'] + nodes[i]["rect"]['height'] for i in cluster)
    
    left_most_point = min(x_values)
    top_most_point = min(y_values)
    right_most_point = max(width_values)
    bottom_most_point = max(height_values)
    
    return {"x": left_most_point, "y": top_most_point, "width": right_most_point - left_most_point, "height": bottom_most_point - top_most_point}

def round_num(num: float, decimal_places: int = 2) -> float:
    factor = 10 ** decimal_places
    return round(num * factor) / factor

def cluster_centrality(window_rect,nodes,cluster: List[int]) -> float:
    bounds = get_cluster_bounds(nodes,cluster)
    center_of_screen = window_rect['width'] / 2
    if bounds["x"] < center_of_screen and bounds["x"] + bounds["width"] > center_of_screen:
        return 0
    if bounds["x"] + bounds["width"] < center_of_screen:
        return center_of_screen - (bounds["x"] + bounds["width"])
    return bounds["x"] - center_of_screen

def percentage_text_share(nodes,cluster: List[int], total_length: int) -> float:
    return round_num((total_text_length(nodes,cluster) / total_length) * 100)


def should_merge_clusters(nodes, cluster_a: List[int], cluster_b: List[int]) -> bool:
    cluster_a_bounds = get_cluster_bounds(nodes, cluster_a)
    cluster_b_bounds = get_cluster_bounds(nodes, cluster_b)
    
    if approximately_equal(cluster_a_bounds["x"], cluster_b_bounds["x"], 40) and approximately_equal(cluster_a_bounds["width"], cluster_b_bounds["width"], 40):
        higher_cluster = cluster_a_bounds if cluster_a_bounds["y"] < cluster_b_bounds["y"] else cluster_b_bounds
        lower_cluster = cluster_a_bounds if cluster_a_bounds["y"] >= cluster_b_bounds["y"] else cluster_b_bounds
        y_gap = lower_cluster["y"] - (higher_cluster["y"] + higher_cluster["height"])
        return approximately_equal(y_gap, 0, 100)
    return False

def find_critical_clusters(window_rect,nodes,clusters: List[List[int]]) -> List[List[int]]:
    # Merge overlapping clusters
    merged_clusters = []
    for i, current_cluster in enumerate(clusters):
        merged = False
        for j, next_cluster in enumerate(clusters[i+1:], start=i+1):
            if should_merge_clusters(nodes, current_cluster, next_cluster):
                current_cluster.extend(next_cluster)
                merged = True
        if not merged:
            merged_clusters.append(current_cluster)
    clusters = merged_clusters
    # Calculate total text length only once
    total_text = total_text_length(nodes, [item for sublist in clusters for item in sublist])

    cluster_with_metrics = [
        {
            "cluster": cluster,
            "centrality": cluster_centrality(window_rect, nodes, cluster),
            "percentage_text_share": percentage_text_share(nodes, cluster, total_text)
        }
        for cluster in clusters
    ]
    
    dominant_cluster = cluster_with_metrics[0]["percentage_text_share"] > 60
    if dominant_cluster:
        return [cluster_with_metrics[0]["cluster"]]
    sorted_clusters = sorted(cluster_with_metrics, key=lambda x: x["percentage_text_share"] * (0.9 ** (x["centrality"] / 100)), reverse=True)
    large_text_share_clusters = [cluster for cluster in sorted_clusters if approximately_equal(cluster["percentage_text_share"], sorted_clusters[0]["percentage_text_share"], 10)]
    total_text_share_of_large_clusters = sum([cluster["percentage_text_share"] for cluster in large_text_share_clusters])
    if total_text_share_of_large_clusters > 60:
        return [cluster["cluster"] for cluster in large_text_share_clusters]
    total_text_share = 0
    critical_clusters = []
    for cluster in sorted_clusters:
        if cluster["percentage_text_share"] < 2:
            continue
        if total_text_share > 60:
            break
        critical_clusters.append(cluster["cluster"])
        total_text_share += cluster["percentage_text_share"]
    if total_text_share < 60:
        return []
    return critical_clusters

def get_attributes(node: WebElement) -> Dict[str, str]:
    """获取元素的特定属性"""
    all_attributes = node.get_property('attributes')
    specific_attributes = {"href", "src", "alt", "title", "class", "id"}
    return {
        attr['name']: attr['value']
        for attr in all_attributes
        if attr['name'] in specific_attributes and attr['value'] is not None
    }

def serialize_children(node: WebElement) -> List[Union[Dict, str]]:
    """序列化元素的所有子节点"""
    children = node.find_elements(By.XPATH, "./*")
    content = [
        serialize_html_element(child) if child.tag_name else child.text
        for child in children
    ]
    return content or [node.text]

def serialize_html_element(node: WebElement, max_depth: int = 5, current_depth: int = 0) -> Dict:
    """序列化HTML元素为字典格式，并同时简化结构"""
    tag_name = node.tag_name.lower()
    attributes = get_attributes(node)

    # 缓存子节点
    if not hasattr(node, '_children'):
        node._children = node.find_elements(By.XPATH, "./*")

    # 检查是否到达最大递归深度
    if current_depth >= max_depth:
        # 返回简化版的信息
        return {
            "tagName": tag_name,
            "attributes": attributes,
            "content": [node.text]
        }

    content = [node.text]

    # 创建简化后的节点结构
    simplified_node = {
        "tagName": tag_name,
        "attributes": attributes,
        "content": content
    }


    return simplified_node
def serialize_node(node: Union[WebElement, str]) -> Union[Dict, str, None]:
    """序列化节点或字符串"""
    if isinstance(node, WebElement):
        return serialize_html_element(node)
    elif isinstance(node, str):
        return node.strip() or None

    
def get_page_metadata(soup:Tag) -> Dict:
    title = soup.title.string if soup.title else ""
    site_name = soup.find("meta", property="og:site_name")["content"] if soup.find("meta", property="og:site_name") else None
    author = soup.find( name="author")["content"] if soup.find(name="author") else None
    description = soup.find(name="description")["content"] if soup.find( name="description") else soup.find(property="og:description")["content"] if soup.find(property="og:description") else None
    created_at = soup.find("meta", property="article:published_time")["content"] if soup.find("meta", property="article:published_time") else soup.find(name="date")["content"] if soup.find(name="date") else None
    updated_at = soup.find("meta", property="article:modified_time")["content"] if soup.find("meta", property="article:modified_time") else None
    return {"title": title, "siteName": site_name, "author": author, "description": description, "createdAt": created_at, "updatedAt": updated_at}

if __name__ == "__main__":
    # Example usage
    #with open('test.html', 'r', encoding='utf-8') as f:
    #    raw_html = f.read()

    import time
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式下运行
    chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速，某些系统/配置需要
    chrome_options.add_argument("--ignore-ssl-errors")  
    chrome_options.add_argument("--no-sandbox")  # 在某些环境中需要
    chrome_options.add_argument("--disable-dev-shm-usage")  # 在某些环境中需要
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.example.com")
    # Wait for the page to load
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    raw_html = driver.page_source
    soup = BeautifulSoup(raw_html, 'html.parser')

    window_rect = driver.get_window_rect()
    readableNodes = get_all_readable_nodes(driver, timeout_seconds = 20)
    print(f"number of readable nodes: {len(readableNodes)}")


    clusters,noise =  cluster_readable_nodes(readableNodes)

    critical_clusters = find_critical_clusters(window_rect,readableNodes,clusters)

    
    cluster_membership = {}
    for cluster in critical_clusters:
        for index in cluster:
            cluster_membership[index] = True

    filtered_nodes = [node for i, node in enumerate(readableNodes) if i in cluster_membership]    

    #输出节点数量
    print(f"number of readable nodes: {len(filtered_nodes)}")
    elements = [serialize_node(node["node"]) for node in filtered_nodes]



    metadata = get_page_metadata(soup)
    #write the output to a pretty json file
    import json
    with open('output.json', 'w') as f:
        json.dump({**metadata, "elements": elements}, f, indent=4)

    # Close the browser
    driver.quit()